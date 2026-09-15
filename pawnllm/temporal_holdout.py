"""Fetch independently compressed Lichess chunks and freeze a later-date holdout.

Sampling is stratified by compressed-file offsets, then by active-player Elo.
This is not a uniform sample of all games in the monthly archive.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import heapq
import io
import json
from pathlib import Path
import random
import re
import struct
import time

import chess
import chess.pgn
import numpy as np
import requests
import zstandard

from pawnllm.chess_data import ShardWriter, encode_board, move_index, outcome_class, speed_class
from pawnllm.chess_model import FORMAT_VERSION

MIB = 1024**2
PZSTD_HEADER = bytes.fromhex('502a4d1804000000')
ZSTD_MAGIC = bytes.fromhex('28b52ffd')
HEADERS = re.compile(rb'^\[([A-Za-z]+) "([^"\r\n]*)"\]$', re.M)
STARTS = re.compile(rb'^\[Event ', re.M)
BANDS = ['600-999','1000-1399','1400-1799','1800-2199','2200-2600']


def digest(data):
    return hashlib.sha256(data).hexdigest()


def priority(game_id, salt):
    return int.from_bytes(hashlib.blake2b((salt+':'+game_id).encode(),digest_size=8).digest(),'big')


def range_bytes(url, start, end, total, etag):
    for attempt in range(10):
        try:
            with requests.get(url, headers={'Range': f'bytes={start}-{end}', 'If-Range': etag},
                              timeout=(15,60), stream=True) as response:
                if response.status_code == 429:
                    time.sleep(max(15, int(response.headers.get('Retry-After','30'))))
                    continue
                response.raise_for_status()
                expected = f'bytes {start}-{end}/{total}'
                if response.status_code != 206 or response.headers.get('Content-Range') != expected:
                    raise ValueError('Server did not honor the exact byte range/version')
                if response.headers.get('ETag') != etag:
                    raise ValueError('Archive changed')
                body=bytearray()
                for chunk in response.iter_content(MIB):
                    body.extend(chunk)
                    if len(body)>end-start+1:
                        raise ValueError('Range response exceeds the requested memory bound')
                if len(body)!=end-start+1:
                    raise ValueError('Incomplete range response')
                return bytes(body)
        except requests.RequestException:
            if attempt == 9:
                raise
            time.sleep(min(30,2**attempt))
    raise RuntimeError('Lichess range requests remained rate-limited after retries')


def fetch_month(month, root, chunks, seed):
    folder = root/month
    folder.mkdir(parents=True,exist_ok=True)
    url=f'https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst'
    head=requests.head(url,timeout=30);head.raise_for_status()
    total=int(head.headers['Content-Length']);etag=head.headers['ETag']
    metadata=dict(month=month,url=url,bytes=total,etag=etag,last_modified=head.headers.get('Last-Modified'),
                  chunks=chunks,seed=seed,sampling='one complete pzstd frame after a seeded random byte offset in each equal-byte stratum')
    manifest=folder/'source.json'
    if manifest.exists() and json.loads(manifest.read_text()) != metadata:
        raise ValueError('Existing source metadata differs; use another output directory')
    manifest.write_text(json.dumps(metadata,indent=2)+'\n')
    rng=random.Random(seed+int(month.replace('-','')))
    # Leave room after the chosen offset for a whole frame; last stratum stops 32 MiB before EOF.
    offsets=[rng.randrange(i*(total-32*MIB)//chunks,(i+1)*(total-32*MIB)//chunks) for i in range(chunks)]
    def fetch(item):
        index,offset=item
        dest=folder/f'{index:04d}.pgn.zst'; meta=dest.with_suffix('.json')
        if dest.exists() and meta.exists():
            record=json.loads(meta.read_text())
            if digest(dest.read_bytes()) != record['sha256']:
                raise ValueError('Cached frame checksum mismatch')
            return record
        data=range_bytes(url,offset,min(total-1,offset+8*MIB-1),total,etag)
        pos=0
        while True:
            pos=data.find(PZSTD_HEADER,pos)
            if pos < 0:
                raise ValueError('No pzstd frame boundary in probe range')
            if data[pos+12:pos+16] == ZSTD_MAGIC:
                break
            pos+=1
        size=struct.unpack('<I',data[pos+8:pos+12])[0]
        if not 0<size<64*MIB:
            raise ValueError('Implausible pzstd frame size')
        start=offset+pos+12;end=start+size-1
        frame=data[pos+12:pos+12+size]
        if len(frame)<size:
            frame+=range_bytes(url,start+len(frame),end,total,etag)
        plain=zstandard.ZstdDecompressor().decompress(frame,max_output_size=128*MIB)
        record=dict(index=index,probe_offset=offset,frame_start=start,frame_end=end,
                    compressed_bytes=len(frame),decompressed_bytes=len(plain),sha256=digest(frame))
        dest.with_suffix('.tmp').write_bytes(frame);dest.with_suffix('.tmp').replace(dest)
        meta.write_text(json.dumps(record,indent=2)+'\n')
        return record
    records=[]
    with ThreadPoolExecutor(max_workers=1) as pool:
        for record in pool.map(fetch,enumerate(offsets)):
            records.append(record)
            if len(records)%8==0:
                print(json.dumps(dict(month=month,downloaded_chunks=len(records),total_chunks=chunks)),flush=True)
    if len({r['frame_start'] for r in records}) != len(records):
        raise ValueError('Duplicate sampled frames')
    (folder/'frames.json').write_text(json.dumps(records,indent=2)+'\n')


def complete_games(frame):
    """Discard partial games at frame boundaries; never concatenate unrelated chunks."""
    starts=[m.start() for m in STARTS.finditer(frame)]
    for begin,end in zip(starts,starts[1:]):
        yield frame[begin:end]


def parse_headers(raw):
    header=raw.split(b'\n\n',1)[0]
    return {k.decode(): v.decode('utf-8') for k,v in HEADERS.findall(header)}


def eligible_header(headers, month, excluded):
    try:
        event=headers['Event']
        if 'Rated Blitz' not in event or headers.get('Variant','Standard')!='Standard':
            return None
        if headers.get('WhiteTitle')=='BOT' or headers.get('BlackTitle')=='BOT':
            return None
        if headers['Result'] not in ('1-0','0-1','1/2-1/2'):
            return None
        if headers.get('Termination') not in ('Normal','Time forfeit'):
            return None
        if speed_class(headers['TimeControl'])!='blitz':
            return None
        white,black=int(headers['WhiteElo']),int(headers['BlackElo'])
        if not (600<=white<=2600 and 600<=black<=2600):
            return None
        stamp=datetime.strptime(headers['UTCDate']+' '+headers['UTCTime'],'%Y.%m.%d %H:%M:%S').replace(tzinfo=timezone.utc)
        if stamp.strftime('%Y-%m')!=month:
            return None
        game_id=headers['Site'].removeprefix('https://lichess.org/')
        if not re.fullmatch('[A-Za-z0-9]{8}',game_id) or game_id in excluded:
            return None
        color=chess.WHITE if priority(game_id,'active-color')%2==0 else chess.BLACK
        rating=white if color else black
        return game_id,color,min(4,(rating-600)//400),stamp
    except (KeyError,ValueError):
        return None


def select_position(game, color, seed):
    """One random eligible position per game, before any clock first falls below 30s."""
    base=int(game.headers['TimeControl'].split('+')[0])
    clocks={chess.WHITE:float(base),chess.BLACK:float(base)}
    board=game.board()
    eligible=[]
    for ply,node in enumerate(game.mainline()):
        after=node.clock()
        # Clock annotations are post-move. Exclude the crossing move as well as
        # every later position; an increment cannot bring a game back into scope.
        if after is None or not np.isfinite(after) or after<30 or min(clocks.values())<30:
            break
        if ply>=20 and board.turn==color and not board.is_game_over(claim_draw=True):
            eligible.append((ply,clocks[board.turn],clocks[not board.turn],float(after)))
        clocks[board.turn]=float(after)
        board.push(node.move)
    if not eligible:
        return None
    return random.Random(seed).choice(eligible)


def build(args):
    output=Path(args.output)
    if output.exists():
        raise ValueError('Output already exists; use a new directory')
    prior=json.loads(Path(args.prior_games).read_text())
    excluded={row['game_id'] for row in prior}
    publication=json.loads(Path(args.publication).read_text())
    uploaded=next(c for c in publication['commits'] if c['commit']==publication['revision'])
    cutoff=datetime.fromisoformat(uploaded['date'])
    writer=ShardWriter(output,4096)
    all_ids=set();stats={};sources={};positions=[];accepted_games=[]
    raw_archive=zstandard.ZstdCompressor(level=6).stream_writer((output/'selected-games.pgn.zst').open('wb'))
    for split,month,quota in [('val',args.val_month,args.val_per_band),('test',args.test_month,args.test_per_band)]:
        folder=Path(args.chunks)/month
        source=json.loads((folder/'source.json').read_text())
        frames=json.loads((folder/'frames.json').read_text())
        sources[split]=dict(source=source,frames=frames)
        heaps=[[] for _ in range(5)]
        seen=Counter();dates=Counter();seen_ids=set()
        # Over-sample candidate games before parsing moves/clocks. If the pool
        # cannot fill quotas, fail rather than silently shrink the benchmark.
        limit=quota*6
        for frame_meta in frames:
            path=folder/f"{frame_meta['index']:04d}.pgn.zst"
            compressed=path.read_bytes()
            if digest(compressed)!=frame_meta['sha256']:
                raise ValueError('Source chunk checksum mismatch')
            plain=zstandard.ZstdDecompressor().decompress(compressed,max_output_size=128*MIB)
            for raw in complete_games(plain):
                seen['scanned_games']+=1
                headers=parse_headers(raw)
                eligible=eligible_header(headers,month,excluded)
                if eligible is None:
                    continue
                game_id,color,band,stamp=eligible
                if stamp<=cutoff:
                    raise ValueError('Game predates pinned Maia publication')
                if game_id in seen_ids:
                    seen['duplicate_games']+=1
                    continue
                seen_ids.add(game_id);dates[stamp.strftime('%Y-%m-%d')]+=1
                seen['eligible_headers']+=1
                rank=priority(game_id,f'temporal-{args.seed}')
                item=(-rank,game_id,raw)
                if len(heaps[band])<limit:
                    heapq.heappush(heaps[band],item)
                elif rank < -heaps[band][0][0]:
                    heapq.heapreplace(heaps[band],item)
        if len(dates)<28:
            raise ValueError('Sample does not cover enough dates across the month')
        counts=Counter();selected_dates=Counter()
        for band,heap in enumerate(heaps):
            for negative_rank,game_id,raw in sorted(heap,reverse=True):
                if counts[BANDS[band]]>=quota:
                    break
                game=chess.pgn.read_game(io.StringIO(raw.decode('utf-8')))
                if game is None or game.errors:
                    seen['invalid_pgn']+=1
                    continue
                info=eligible_header(game.headers,month,excluded)
                _,color,_,stamp=info
                selected=select_position(game,color,priority(game_id,'position-'+str(args.seed)))
                if selected is None:
                    seen['no_clock_safe_position']+=1
                    continue
                ply,active_clock,opponent_clock,after_clock=selected
                board=game.board();moves=list(game.mainline_moves())
                for move in moves[:ply]:
                    board.push(move)
                human=moves[ply]
                ratings=np.array([int(game.headers['WhiteElo']),int(game.headers['BlackElo'])],dtype=np.float32)
                if not board.turn:
                    ratings=ratings[::-1].copy()
                pieces,rules=encode_board(board,8)
                indices=[move_index(m,board.turn) for m in board.legal_moves]
                legal=np.full(256,-1,dtype=np.int16);legal[:len(indices)]=indices
                sample=(pieces,rules,legal,move_index(human,board.turn),outcome_class(game.headers['Result'],board.turn),ratings,np.int16(ply))
                if game_id in all_ids or game_id in excluded:
                    raise ValueError('Cross-split or prior-game leakage')
                all_ids.add(game_id)
                writer.add(split,[sample])
                positions.append(dict(split=split,game_id=game_id,ply=ply,moves=[m.uci() for m in moves[:ply]],
                    human_move=human.uci(),rating=int(ratings[0]),opponent_rating=int(ratings[1]),fen=board.fen(),
                    utc=stamp.isoformat(),time_control=game.headers['TimeControl'],active_clock_before=active_clock,
                    opponent_clock_before=opponent_clock,active_clock_after=after_clock))
                accepted_games.append(dict(game_id=game_id,split=split,utc=stamp.isoformat(),rating_band=BANDS[band]))
                raw_archive.write(raw)
                counts[BANDS[band]]+=1;selected_dates[stamp.strftime('%Y-%m-%d')]+=1
            if counts[BANDS[band]]!=quota:
                raise ValueError(f'Insufficient {split}/{BANDS[band]}: {counts[BANDS[band]]}/{quota}; fetch more chunks')
        writer.flush(split)
        stats[split]=dict(scan=dict(seen),positions_by_rating=dict(counts),games_by_date=dict(selected_dates),candidate_dates=dict(dates))
        print(json.dumps(dict(split=split,counts=dict(counts),scanned=seen['scanned_games'])),flush=True)
    raw_archive.close()
    if max(r['utc'] for r in positions if r['split']=='val')>=min(r['utc'] for r in positions if r['split']=='test'):
        raise ValueError('Validation/test dates overlap')
    (output/'positions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in positions))
    (output/'games.json').write_text(json.dumps(accepted_games,indent=2)+'\n')
    (output/'excluded-prior-games.json').write_text(json.dumps(prior)+'\n')
    (output/'maia-publication.json').write_text(json.dumps(publication,indent=2)+'\n')
    manifest=dict(format_version=FORMAT_VERSION,args=dict(history=8,**vars(args)),rating_conditioned=True,
        positions=dict(writer.counts),split='validation July 2026, test August 2026; one position per game; no train split',
        stats=stats,sources=sources,maia_publication=publication,excluded_prior_games=len(excluded),
        sampling='Stratified compressed-byte chunks, then smallest seeded game hashes per active-player Elo band; one eligible position per game',
        filters='Rated standard blitz; both 600-2600; no BOT; normal/time-forfeit with known result; ply>=20; before first missing/sub-30s clock',
        limitation='Post-publication games cannot be examples in the pinned earlier weights, but recurring positions and human players can overlap.',
        status='Frozen holdout; no model inference, calibration fitting, or training performed on these sets',
        future_training_rule='Never train on July/August 2026 games or any game ID in this holdout. Use validation for selection/calibration; reserve test for final evaluation.')
    hashes={str(p.relative_to(output)):digest(p.read_bytes()) for p in output.rglob('*') if p.is_file()}
    manifest['files_sha256']=hashes
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(output=str(output),positions=manifest['positions'],excluded=len(excluded))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('fetch');f.add_argument('--month',required=True);f.add_argument('--output',default='/mnt/archive/lichess/temporal-chunks')
    f.add_argument('--chunks',type=int,default=96);f.add_argument('--seed',type=int,default=20260915)
    b=sub.add_parser('build');b.add_argument('--chunks',default='/mnt/archive/lichess/temporal-chunks')
    b.add_argument('--output',default='/mnt/archive/lichess/temporal-holdout-2026')
    b.add_argument('--val-month',default='2026-07');b.add_argument('--test-month',default='2026-08')
    b.add_argument('--val-per-band',type=int,default=1000);b.add_argument('--test-per-band',type=int,default=2000)
    b.add_argument('--prior-games',default='runs/temporal-holdout/prior-games.json')
    b.add_argument('--publication',default='runs/temporal-holdout/maia-publication.json');b.add_argument('--seed',type=int,default=20260915)
    args=p.parse_args()
    if args.command=='fetch':
        fetch_month(args.month,Path(args.output),args.chunks,args.seed)
    else:
        build(args)


if __name__=='__main__':
    main()
