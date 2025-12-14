import os
import argparse
from dotenv import load_dotenv
from google import genai
from google.genai import types

from prompts import system_prompt
from functions.get_files_info import schema_get_files_info
from functions.write_file_content import schema_write_file
from functions.run_python_file import schema_run_python_file
from functions.get_file_content import schema_get_file_content

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if api_key is None:
    raise ValueError("API key is required")

available_functions = types.Tool(
    function_declarations=[schema_get_files_info, schema_run_python_file, schema_write_file, schema_get_file_content],
)
def main():
    parser = argparse.ArgumentParser(description="bootagentai")
    parser.add_argument("user_prompt", type=str, help="user prompt")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    client = genai.Client(api_key=api_key)

    messages = [types.Content(role="user", parts=[types.Part(text=args.user_prompt)])]

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=messages,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[available_functions],
        ),
    )

    if args.verbose:
        print(f"User prompt: {args.user_prompt}")
        print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
        print(f"Response tokens: {response.usage_metadata.candidates_token_count}")

    for function_call in response.function_calls:
        if function_call:
            print(f"Calling function: {function_call.name}({function_call.args})")
        else:
            print(response.text)

if __name__ == "__main__":
    main()
