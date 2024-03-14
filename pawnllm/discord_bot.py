import os
import asyncio
import discord
from discord.ext import tasks, commands
from dotenv import load_dotenv
import torch

load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DISCORD_CHANNEL = os.getenv("DISCORD_CHANNEL")
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=".",intents=intents)


checkpoint_path = os.path.join("../data", "checkpoint.tar")

discord_text="None"

@bot.event
async def on_ready():
    channel = bot.get_channel(int(DISCORD_CHANNEL))
    printMessage.start()
    await channel.send("Training Tracker Bot is on")

@tasks.loop(minutes=60)
async def printMessage():
    if not os.path.isfile(checkpoint_path):
        return
    checkpoint = torch.load(checkpoint_path)
    step_start = checkpoint["epoch"] + 1
    loss = checkpoint["loss"]
    discord_text = f"current checkpoint = step:{step_start}\tfinal_loss={loss.item()}"
    channel = bot.get_channel(int(DISCORD_CHANNEL))
    await channel.send(discord_text)

async def main():
    await bot.start(DISCORD_TOKEN)
asyncio.run(main())