from langchain_ibm import ChatWatsonx
from dotenv import load_dotenv
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods

parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 100
}

load_dotenv()

chat = ChatWatsonx(model_id="ibm/granite-13b-chat-v2",
                   project_id="22900986-b8a0-4242-9fdc-1de6ac220c4a",
                   params=parameters)

result = chat.invoke("What is 2 + 2")

print(result.content)
