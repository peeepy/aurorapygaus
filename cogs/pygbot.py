import re
import json
import requests
import discord
from discord.ext import commands

endpoint = "https://beast-sms-meters-module.trycloudflare.com/"


class Chatbot:
    def __init__(self, char_filename):
        with open(char_filename, "r") as f:
            data = json.load(f)
            self.char_title = data["char_name"]
            self.char_name = data["name"]
            self.char_persona = data["char_persona"]
            self.char_greeting = data["char_greeting"]
            self.world_scenario = data["world_scenario"]
            self.example_dialogue = data["example_dialogue"]

        self.history = [
            f"{self.char_title}\n{self.char_name}'s Persona: {self.char_persona}\nWorld Scenario: {self.world_scenario}\n{self.example_dialogue}\n{self.char_name}: {self.char_greeting}\n"]
        self.prompt = None

    def generate_response(self):
        # Generate response based on the user's message and the prompt
        response = requests.post(f"{endpoint}/api/v1/generate", json=self.prompt)
        results = response.json()['results']
        # extract the correct bot reponse from the large json of information
        text = results[0]['text']
        parts = re.split(r'\n[a-zA-Z]', text)[:1]
        response_text = parts[0][1:]
        if len(self.history) > 100:
            self.history = self.history[-100:]
        return response_text

    def add_message(self, speaker, message_content):
        self.history.append(f"{speaker}: {message_content}\n")

    def save_conversation(self, message, message_content, bot, count):
        message_content = message.content.replace(f"<@{bot.user.id}>", "").strip()
        self.add_message(message.author.name, message_content)
        print(f"{message.author.name}: {message_content}")
        if count > 4:
            self.history = [f"{self.char_title}\n{self.char_name}'s Persona: {self.char_persona}\n"]
        self.prompt = {"prompt": '\n'.join(self.history) + f"{self.char_name}:", "use_story": False,
                       "use_memory": False,
                       "use_authors_note": False, "use_world_info": False, "max_context_length": 1400,
                       "max_length": 70, "rep_pen": 1.05, "rep_pen_range": 800, "rep_pen_slope": 0.9,
                       "temperature": 0.5, "tfs": 0.9, "top_a": 0, "top_k": 0, "top_p": 0.9,
                       "typical": 1.0, "sampler_order": [6, 0, 1, 2, 3, 4, 5], "frmttriminc": True, "frmtrmblln": True}
        print(self.prompt)
        bot_response = self.generate_response()
        self.add_message(self.char_name, bot_response)
        print(f"{self.char_name}: {bot_response}")
        count += 1
        print(f"count: {count}")
        return bot_response


# Here we name the cog and create a new class for the cog.
class ChatbotCog(commands.Cog, name="chatbot"):
    def __init__(self, bot):
        self.bot = bot
        self.chatbot = Chatbot("Tensorsama.json")
        self.count = 0

    @commands.command(name="chat")
    async def chat_command(self, message: discord.Message, message_content, bot, count) -> None:
        response = self.chatbot.save_conversation(message, message_content, bot, count)
        return response


async def setup(bot):
    await bot.add_cog(ChatbotCog(bot))

