import os
import re
import logging
import aiohttp
import asyncio
import requests
import json

from dotenv import load_dotenv

from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession
from livekit.agents.voice.room_io import RoomInputOptions, RoomOutputOptions
from livekit.plugins import openai, deepgram, cartesia, noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("assort-health")

SYSTEM_PROMPT = """
You are a helpful and persistent healthcare booking agent.

Your task is to gather **all necessary patient details** required for scheduling a medical appointment.
The conversation MUST continue until **every required piece of information** is collected and confirmed.

You must collect the following:
1. **Full name** (e.g., "Jane Doe")
2. **Date of birth** (format: YYYY-MM-DD or "June 2nd, 1980")
3. **Insurance information:**
   - Payer name (e.g., "Blue Cross Blue Shield")
   - Insurance ID (e.g., "XYZ123456789")
4. **Referring physician**, if any (e.g., "Dr. Thompson")
5. **Chief complaint or reason for visit** (e.g., "persistent lower back pain")
6. **Street address** (validated using `validate_address`)
7. **City, state, and ZIP code**
8. **Contact details:**
   - Phone number (e.g., "(123) 456-7890")
   - Email address `email_address` (e.g., "jane.doe@example.com")
9. **Preferred provider and appointment time** (choose from `get_available_providers`)
   - Provider name (store as `provider_name`)
   - Appointment time (store as `appointment_time`)
10. **Appointment date** (e.g., "2025-05-15")

For **every individual field collected**, you MUST immediately call:
store_info("<key>", "<value>")
Example:
store_info("full_name", "Jane Doe")

After all fields are collected, you must call:
schedule_appointment(<all stored fields>)

Important response guidelines:
- Your replies will be converted to voice and spoken out loud.
- Keep your responses concise and easy to understand in spoken form.
- Do NOT use markdown, bullet points, or formatted text in your replies.

General guardrails:
- Maintain a professional, calm, and friendly tone at all times.
- Never invent information, names, or data not provided by the user or API.
- Do not offer medical advice or opinions — you are a scheduling assistant only.
- Use only the tools and functions you have been provided (e.g., `store_info`, `validate_address`, `get_available_providers`, `schedule_appointment`).
- If the user says something unrelated to appointment scheduling, politely steer the conversation back to collecting the required information.

You may end the conversation **only after** all required details have been collected and an appointment has been scheduled. Do **not** end the conversation early, even if the user attempts to, until the booking is complete.
"""


class BookingAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)
        self.storage = {}

    async def llm_node(self, chat_ctx, tools, model_settings=None):
        activity = self._get_activity_or_raise()
        assert activity.llm is not None, "llm_node called but no LLM node is available"

        def remove_markdown(text: str) -> str:
            text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)      
            text = re.sub(r"\*(.*?)\*", r"\1", text)           
            text = re.sub(r"_(.*?)_", r"\1", text)             
            text = re.sub(r"`(.*?)`", r"\1", text)           
            text = re.sub(r"#+\s*(.*?)\n?", r"\1\n", text)       
            text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)     
            return text

        async def process_stream():
            async with activity.llm.chat(chat_ctx=chat_ctx, tools=tools, tool_choice=None) as stream:
                async for chunk in stream:
                    if chunk is None:
                        continue

                    content = getattr(chunk.delta, 'content', None) if hasattr(chunk, 'delta') else str(chunk)
                    if content is None:
                        yield chunk
                        continue

                    processed_content = remove_markdown(content)

                    if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'content'):
                        chunk.delta.content = processed_content
                    else:
                        chunk = processed_content

                    yield chunk

        return process_stream()
    

    @function_tool
    async def validate_address(self, street: str, city: str, state: str, zip_code: str) -> dict:
        """
        Validates a patient's address using the Smarty US Street Address API.

        Args:
            street: Street address (e.g., "1600 Amphitheatre Pkwy")
            city: City (e.g., "Mountain View")
            state: State abbreviation (e.g., "CA")
            zip_code: ZIP code (optional; improves match accuracy)

        Returns:
            A dictionary with:
                - 'valid': True if the address is valid (non-empty response), otherwise False
                - 'normalized': the first validated address result (if any), or an empty dict
        """
        from urllib.parse import urlencode

        auth_id = os.getenv("SMARTY_AUTH_ID")
        auth_token = os.getenv("SMARTY_AUTH_TOKEN")

        params = {
            "auth-id": auth_id,
            "auth-token": auth_token,
            "street": street,
            "city": city,
            "state": state,
            "zipcode": zip_code,
            "candidates": 1,
        }

        base_url = "https://us-street.api.smarty.com/street-address"
        query_string = urlencode(params)
        url = f"{base_url}?{query_string}"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    logging.info(data)
                    is_valid = len(data) > 0
                    normalized = data[0] if is_valid else {}
                    return {"valid": is_valid, "normalized": normalized}
                else:
                    raise Exception(f"Address validation failed: HTTP {response.status}")


    @function_tool
    async def get_available_providers(self) -> list:
        """
        Returns list of available doctors and times.

        Returns:
            A list of providers with name, specialty, and available appointment times.
        """
        return [
            {"name": "Dr. Alice Smith", "specialty": "Primary Care", "times": ["2025-05-15T09:00", "2025-05-15T10:00"]},
            {"name": "Dr. Bob Jones", "specialty": "Cardiology", "times": ["2025-05-16T11:00", "2025-05-16T13:00"]},
            {"name": "Dr. Carol Lee", "specialty": "Dermatology", "times": ["2025-05-17T14:00", "2025-05-17T15:30"]},
        ]

    @function_tool
    async def store_info(self, key: str, value: str) -> dict:
        """
        Stores a key-value pair of patient data.

        Args:
            key: The type of information being stored (e.g., 'full_name', 'date_of_birth', etc.).
            value: The actual value provided by the patient.

        Valid keys:
            - full_name
            - date_of_birth
            - insurance_payer
            - insurance_id
            - referral
            - chief_complaint
            - address_street
            - address_city
            - address_state
            - address_zip
            - contact_phone
            - provider_name
            - appointment_date
            - appointment_time
            - email_address

        Returns:
            A dictionary confirming storage of the data.
        """
        logging.info(f"Storing info: {key} = {value}")
        self.storage[key] = value
        return {key: value}
    
    @function_tool
    async def schedule_appointment(
        self,
        full_name: str,
        provider_name: str,
        appointment_date: str,
        appointment_time: str,
    ) -> dict:
        """
        Schedules the appointment using the patient's name, chosen provider, date, and time.

        Args:
            full_name: Full name of the patient (e.g., "Jane Doe").
            provider_name: Selected provider's full name (e.g., "Dr. Alice Smith").
            appointment_date: Appointment date (e.g., "2025-05-15").
            appointment_time: Selected appointment time (e.g., "09:00").

        Returns:
            A dictionary confirming the scheduled appointment:
                - 'status': Always "confirmed"
                - 'confirmation_number': Mock confirmation number
                - 'provider': Name of the selected provider
                - 'date': Appointment date
                - 'time': Appointment time
        """
        confirmation_number = "APT-" + str(abs(hash(full_name + appointment_date + appointment_time)))[:6]
        logging.info(f"Scheduled appointment for {full_name} on {appointment_date} at {appointment_time} with {provider_name}")
        return {
            "status": "confirmed",
            "confirmation_number": confirmation_number,
            "provider": provider_name,
            "date": appointment_date,
            "time": appointment_time,
        }


async def send_confirmation_email(patient_email: str, appointment_details: dict) -> dict:
    """
    Sends a confirmation email with patient and appointment info.

    Args:
        patient_email: Patient's email address.
        appointment_details: All stored appointment details.

    Returns:
        Dict with email send status.
    """
    try:
        zapier_url = "https://hooks.zapier.com/hooks/catch/22883865/2nct4ys/"

        full_name = appointment_details.get("full_name", "")
        provider = appointment_details.get("provider_name", "Unknown provider")
        date = appointment_details.get("appointment_date", "Unknown date")
        time = appointment_details.get("appointment_time", "Unknown time")
        
        message = (
            f"Hello {full_name}\n\n"
            f"Your appointment has been confirmed with {provider} on {date} at {time}.\n\n"
            f"Please arrive 10 minutes early. If you have any questions, feel free to contact us.\n\n"
            f"Thank you!"
        )

        zapier_payload = json.dumps({
            "send_to": patient_email,
            "message": message
        })
        headers = {'Content-Type': 'application/json'}
        response = requests.post(zapier_url, headers=headers, data=zapier_payload)
        response.raise_for_status()

        logger.info(f"Confirmation email sent to {patient_email}")
        return {"status": "sent", "email": patient_email}

    except Exception as e:
        logger.warning(f"Zapier webhook failed: {e}")
        return {"status": "failed", "error": str(e)}


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    ctx.log_context_fields = {"room_name": ctx.room.name}

    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=openai.LLM(model="gpt-4o"),
        tts=cartesia.TTS(speed=0.8, voice="694f9389-aac1-45b6-b726-9d9369183238"),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    agent = BookingAgent()

    @ctx.room.on("participant_disconnected")
    def on_session_close():
        async def finalize():
            logger.info("Session closed. Final collected info:")
            for key, value in agent.storage.items():
                logger.info(f"{key}: {value}")

            email = agent.storage.get("email_address")
            if email:
                try:
                    await send_confirmation_email(email, agent.storage)
                    logger.info("Confirmation email sent.")
                except Exception as e:
                    logger.error(f"Failed to send confirmation email: {e}")
            else:
                logger.warning("No email found in stored data. Skipping confirmation email.")

        asyncio.create_task(finalize())

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVCTelephony()),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )
    await session.generate_reply(
    instructions="Greet the caller, introduce yourself as a voice assistant who will help schedule a medical appointment, and ask if they are ready to begin."
)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
