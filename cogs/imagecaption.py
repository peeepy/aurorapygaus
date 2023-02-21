import discord
from discord.ext import commands
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image, ImageTk 
from io import BytesIO
import requests

class ImageCaptionCog(commands.Cog, name="image_caption"):
    def __init__(self, bot):
        self.bot = bot
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

    # Define the "chat" command that will be executed by the user.
    @commands.command(name="image_comment")
    async def image_comment(self, message: discord.Message, message_content) -> None:
        message.author.name
        # Download the image from the message and convert it to a PIL image
        image_url = message.attachments[0].url
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        inputs = self.processor(image, return_tensors="pt").to("cuda")
        out = self.model.generate(**inputs, max_length=50)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        message_content = f"{message.content} [{message.author.name} posts a picture of {caption}]"
        return message_content


async def setup(bot):
    await bot.add_cog(ImageCaptionCog(bot))