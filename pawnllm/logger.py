import os
import asyncio
import discord
from dotenv import load_dotenv

load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DISCORD_CHANNEL = int(os.getenv("DISCORD_CHANNEL"))


async def discord_async_connect(client):
    await client.login(DISCORD_TOKEN)
    await client.connect()


async def discord_async_send_and_close(client, message):
    await client.wait_until_ready()
    channel = client.get_channel(DISCORD_CHANNEL)
    if channel:
        await channel.send(message)
    await client.close()


async def discord_log_async(client, message):
    await asyncio.gather(discord_async_connect(client), discord_async_send_and_close(client, message))


def discord_log(message):
    client = discord.Client(intents=discord.Intents.default())
    asyncio.run(discord_log_async(client, message))


if __name__ == "__main__":
    discord_log("```hello world```")
    discord_log("```bbbbb```")
