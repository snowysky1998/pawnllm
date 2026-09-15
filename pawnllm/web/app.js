'use strict';
const $ = id => document.getElementById(id);
const names = {p:'pawn',n:'knight',b:'bishop',r:'rook',q:'queen',k:'king'};
let game = null, preview = null, selected = null, busy = false, flipped = false;
const view = () => preview || game;
const percent = value => `${(100*value).toFixed(1)}%`;

function error(message) { $('error').textContent = message; $('error').hidden = !message; }
function setBusy(value) {
  busy = value;
  for (const id of ['new','white-elo','black-elo','flip','depth','analysis-back','analysis-live']) $(id).disabled = value;
  $('undo').disabled = value || !game || game.moves.length < 1 || Boolean(preview);
  $('board').setAttribute('aria-busy', String(value));
  $('status-dot').classList.toggle('busy', value);
  $('candidates').classList.toggle('loading', value);
}
function render() {
  if (!game) return;
  const state = view();
  $('color-tag').textContent = 'You control both sides';
  $('status').textContent = preview && !state.over ? `Exploring · ${state.analysis.turn === 'white' ? 'White' : 'Black'} to move` : state.status;
  $('preview-banner').hidden = !preview;
  $('preview-path').textContent = preview ? `After ${preview.moves.length} half-moves · ${preview.sans.slice(-6).join(' ') || 'Starting position'}` : '';
  const reverse = flipped;
  const ranks = reverse ? [0,1,2,3,4,5,6,7] : [7,6,5,4,3,2,1,0];
  const files = reverse ? [7,6,5,4,3,2,1,0] : [0,1,2,3,4,5,6,7];
  const destinations = new Set(state.legal.filter(m => m.startsWith(selected || '!')).map(m => m.slice(2,4)));
  const last = state.moves.at(-1) || '';
  $('board').replaceChildren();
  for (const [row,rank] of ranks.entries()) for (const [col,file] of files.entries()) {
    const square = 'abcdefgh'[file] + (rank+1), piece = state.pieces[square];
    const button = document.createElement('button');
    button.className = 'square' + ((rank+file)%2 === 0 ? ' dark' : '');
    button.dataset.square = square;
    button.classList.toggle('occupied', Boolean(piece));
    button.classList.toggle('selected', selected === square);
    button.classList.toggle('target', destinations.has(square));
    button.classList.toggle('last', last.slice(0,2) === square || last.slice(2,4) === square);
    button.classList.toggle('check', state.check === square);
    const pieceName = piece ? `${piece === piece.toUpperCase() ? 'White' : 'Black'} ${names[piece.toLowerCase()]}` : 'empty';
    button.setAttribute('aria-label', `${square}, ${pieceName}`);
    button.setAttribute('aria-pressed', String(selected === square));
    button.onclick = () => clickSquare(square);
    if (piece) { const img = document.createElement('img'); img.src = `/piece/${piece}.svg`; img.alt = ''; img.draggable = false; button.append(img); }
    if (col === 0) { const label = document.createElement('span'); label.className='coordinate rank'; label.textContent=rank+1; button.append(label); }
    if (row === 7) { const label = document.createElement('span'); label.className='coordinate file'; label.textContent='abcdefgh'[file]; button.append(label); }
    $('board').append(button);
  }
  $('move-count').textContent = `${game.moves.length} played`;
  $('history').replaceChildren();
  if (!game.sans.length) { const text=document.createElement('p'); text.className='empty'; text.textContent='Move either side to start your analysis.'; $('history').append(text); }
  for (let i=0; i<game.sans.length; i+=2) {
    const row=document.createElement('div'); row.className='move-row';
    const number=document.createElement('span'); number.textContent=`${i/2+1}.`; row.append(number);
    for (const ply of [i,i+1]) {
      const cell=document.createElement(game.sans[ply] ? 'button' : 'span'); cell.textContent=game.sans[ply] || '';
      if (game.sans[ply]) { cell.title='Inspect the position after this move'; cell.onclick=()=>requestGame(game.moves.slice(0,ply+1),'preview'); }
      row.append(cell);
    }
    $('history').append(row);
  }
  $('history').scrollTop = $('history').scrollHeight;
  renderAnalysis();
  renderArrows();
}
function markMove(move) {
  document.querySelectorAll('.square.hint').forEach(square => square.classList.remove('hint'));
  if (move) for (const square of [move.slice(0,2),move.slice(2,4)]) document.querySelector(`[data-square="${square}"]`)?.classList.add('hint');
}
function renderAnalysis() {
  const state = view(), analysis = state.analysis;
  if (!analysis) return;
  $('model-name').textContent = `${analysis.model === 'human' ? 'Human' : analysis.model === 'enhanced' ? 'Enhanced' : 'Pilot'} · ${(analysis.parameters/1e6).toFixed(1)}M`;
  $('depth').value = analysis.depth;
  $('white-elo').value = analysis.white_elo;
  $('black-elo').value = analysis.black_elo;
  const labels = ['White win','Draw','Black win'];
  $('wdl-labels').replaceChildren();
  analysis.wdl.forEach((probability,i) => {
    $('wdl-bar').children[i].style.width=percent(probability);
    const label=document.createElement('span'); label.textContent=`${labels[i]} ${percent(probability)}`; $('wdl-labels').append(label);
  });
  $('wdl-bar').setAttribute('aria-label', analysis.wdl.map((p,i)=>`${labels[i]} ${percent(p)}`).join(', '));
  $('policy-side').textContent = analysis.terminal ? 'Game over · exact result' : `${analysis.turn === 'white' ? 'White' : 'Black'} · ${analysis.turn === 'white' ? analysis.white_elo : analysis.black_elo} Elo · legal moves`;
  $('policy-coverage').textContent = analysis.terminal ? 'The result above is exact; no network evaluation is needed.' : `Top ${analysis.candidates.length} of ${analysis.legal_count} legal moves · ${percent(analysis.shown_probability)} of policy mass. Policy percentages use all legal moves.`;
  $('candidates').replaceChildren();
  for (const candidate of analysis.candidates) {
    const row = document.createElement('div'); row.className='candidate';
    const button = document.createElement('button'); button.className='candidate-select';
    button.setAttribute('aria-label', `Explore ${candidate.san}, policy ${percent(candidate.probability)}`);
    const move=document.createElement('strong'); move.textContent=candidate.san;
    const track=document.createElement('span'); track.className='policy-track';
    const fill=document.createElement('span'); fill.style.width=percent(candidate.probability); track.append(fill);
    const probability=document.createElement('span'); probability.textContent=percent(candidate.probability);
    button.append(move,track,probability);
    button.onclick=()=>requestGame([...state.moves,candidate.uci],'preview');
    button.onmouseenter=()=>markMove(candidate.uci); button.onmouseleave=()=>markMove(null);
    button.onfocus=()=>markMove(candidate.uci); button.onblur=()=>markMove(null);
    row.append(button);
    const detail=document.createElement('div'); detail.className='candidate-detail';
    const line=document.createElement('div'); line.className='candidate-line';
    candidate.line.forEach((ply,i)=>{
      const step=document.createElement('button'); step.textContent=ply.san; step.title=`Explore through ${ply.san}`;
      step.onclick=()=>requestGame([...state.moves,...candidate.line.slice(0,i+1).map(m=>m.uci)],'preview');
      line.append(step);
    });
    const value=document.createElement('span'); value.className='line-value';
    value.textContent=`White ${percent(candidate.white_score)}${candidate.terminal ? ' · final' : ''}`;
    value.title=`White expected points after ${candidate.line.length} half-moves. White win ${percent(candidate.leaf_wdl[0])}, draw ${percent(candidate.leaf_wdl[1])}, Black win ${percent(candidate.leaf_wdl[2])}.`;
    detail.append(line,value); row.append(detail); $('candidates').append(row);
  }
}
async function requestGame(moves, mode='main') {
  if (busy) return;
  selected = null;
  error('');
  setBusy(true);
  $('status').textContent = 'Predicting human moves…';
  try {
    const endpoint = '/api/analyze';
    const response = await fetch(endpoint, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({moves,depth:Number($('depth').value),white_elo:Number($('white-elo').value),black_elo:Number($('black-elo').value)})});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Could not update the game.');
    if (mode === 'preview' || (mode === 'refresh' && preview)) preview=result;
    else {
      game = result; preview=null;

    }
    try { localStorage.setItem('pawnllm-analysis', JSON.stringify({moves:game.moves,white_elo:result.analysis.white_elo,black_elo:result.analysis.black_elo})); } catch (_) { /* Storage is optional. */ }
    render();
  } catch (err) {
    error(err.message || 'Could not reach the engine. Check that the server is running.');
    if (game) render();
    else $('status').textContent = 'Could not load game';
  } finally { setBusy(false); }
}
function clickSquare(square) {
  const state = view();
  if (busy || !state || state.over) return;
  const choices = selected ? state.legal.filter(move => move.startsWith(selected+square)) : [];
  if (choices.length) {
    if (choices.length === 1) submitMove(choices[0]);
    else showPromotion(choices);
    return;
  }
  selected = selected === square ? null : state.legal.some(move=>move.startsWith(square)) ? square : null;
  render();
  document.querySelector(`[data-square="${square}"]`)?.focus({preventScroll:true});
}
function submitMove(move) {
  $('promotion').close();
  requestGame([...view().moves,move], preview ? 'preview' : 'main');
}
function showPromotion(choices) {
  const options=$('promotion').querySelector('.promotion-options'); options.replaceChildren();
  for (const type of ['q','r','b','n']) {
    const move=choices.find(m=>m.endsWith(type)); if (!move) continue;
    const button=document.createElement('button'); button.setAttribute('aria-label', `Promote to ${names[type]}`);
    const img=document.createElement('img'); img.src=`/piece/${view().analysis.turn==='white'?type.toUpperCase():type}.svg`; img.alt=names[type];
    button.append(img); button.onclick=()=>submitMove(move); options.append(button);
  }
  $('promotion').showModal();
}
function returnLive() { if (busy) return; requestGame(game.moves,'main'); }
$('cancel-promotion').onclick=()=>$('promotion').close();
$('new').onclick=()=>requestGame([]);
$('undo').onclick=()=>{ if(game && game.moves.length) requestGame(game.moves.slice(0,-1)); };
$('flip').onclick=()=>{flipped=!flipped; render();};
$('depth').onchange=()=>{if(view()) requestGame(view().moves,'refresh');};
$('analysis-live').onclick=returnLive;
$('analysis-back').onclick=()=>{if(!preview || !preview.moves.length || preview.moves.length===game.moves.length+1) returnLive(); else requestGame(preview.moves.slice(0,-1),'preview');};
for (const id of ['white-elo','black-elo']) {
  for (let elo=600;elo<=2600;elo+=100) {const option=document.createElement('option');option.value=elo;option.textContent=elo;$(id).append(option);}
  $(id).value=1500;
  $(id).onchange=()=>{if(view()) requestGame(view().moves,'refresh');};
}
let saved;
try { saved=JSON.parse(localStorage.getItem('pawnllm-analysis')); } catch (_) { saved=null; }
for (const [id,key] of [['white-elo','white_elo'],['black-elo','black_elo']]) {
  if (Number.isInteger(saved?.[key]) && saved[key]>=600 && saved[key]<=2600 && saved[key]%100===0) $(id).value=saved[key];
}
requestGame(Array.isArray(saved?.moves) ? saved.moves : []);

function renderArrows() {
  const svg=$('arrows'), analysis=view()?.analysis;
  svg.replaceChildren();
  if (!analysis) return;
  const ns='http://www.w3.org/2000/svg';
  const colors=['#d69e19','#278a9a','#a363b6'];
  const defs=document.createElementNS(ns,'defs');
  colors.forEach((color,i)=>{
    const marker=document.createElementNS(ns,'marker');
    for (const [key,value] of Object.entries({id:`arrow-${i}`,viewBox:'0 0 10 10',refX:'8',refY:'5',markerWidth:'3.3',markerHeight:'3.3',orient:'auto-start-reverse'})) marker.setAttribute(key,value);
    const tip=document.createElementNS(ns,'path');tip.setAttribute('d','M 0 0 L 10 5 L 0 10 z');tip.setAttribute('fill',color);marker.append(tip);defs.append(marker);
  });
  svg.append(defs);
  const point=square=>{
    const file='abcdefgh'.indexOf(square[0]), rank=Number(square[1])-1;
    return flipped ? [(7-file)*100+50,rank*100+50] : [file*100+50,(7-rank)*100+50];
  };
  // Draw lower-ranked arrows first; offset coincident promotion paths so all remain visible.
  const top=analysis.candidates.slice(0,3);
  for (let i=top.length-1;i>=0;i--) {
    const row=top[i], from=point(row.uci.slice(0,2)), to=point(row.uci.slice(2,4));
    const dx=to[0]-from[0],dy=to[1]-from[1],length=Math.hypot(dx,dy);
    const same=top.filter(x=>x.uci.slice(0,4)===row.uci.slice(0,4));
    const offset=same.length>1 ? (same.indexOf(row)-(same.length-1)/2)*14 : 0;
    const ox=-dy/length*offset,oy=dx/length*offset;
    const path=document.createElementNS(ns,'path');
    path.setAttribute('d',`M ${from[0]+ox} ${from[1]+oy} L ${to[0]-dx/length*14+ox} ${to[1]-dy/length*14+oy}`);
    path.setAttribute('stroke',colors[i]);path.setAttribute('stroke-width',i===0?'11':'8');
    path.setAttribute('stroke-linecap','round');path.setAttribute('fill','none');path.setAttribute('opacity','.82');
    path.setAttribute('marker-end',`url(#arrow-${i})`);path.dataset.rank=i+1;path.dataset.move=row.uci;
    const title=document.createElementNS(ns,'title');title.textContent=`${i+1}. ${row.san}: ${percent(row.probability)}`;path.append(title);svg.append(path);
  }
  svg.setAttribute('aria-label',top.map((row,i)=>`${i+1}. ${row.san}, ${percent(row.probability)}`).join('; ') || 'No legal moves');
}
