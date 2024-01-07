import os
from typing import List

from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential


class KeywordsExtractor:
    NON_KEYWORDS = [
        "Location",
        "Country",
        "Province",
        "Description",
        "Details",
        "Designation",
        "Points",
        "Price",
        "Region_1",
        "Taster_name",
        "handle",
        "Variety",
        "Winery",
    ]

    def __init__(self):
        self._key = os.environ.get("AZURE_KEY_PHRASE_KEY")
        self._endpoint = os.environ.get("AZURE_KEY_PHRASE_ENDPOINT")
        self._client = self._authenticate_client()

    def _authenticate_client(self):
        ta_credentials = AzureKeyCredential(self._key)
        text_analytics_client = TextAnalyticsClient(
            endpoint=self._endpoint, credential=ta_credentials
        )
        return text_analytics_client

    def extract(self, text: str) -> List[str]:
        try:
            response = self._client.extract_key_phrases(documents=[text])[0]
            keywords = self._clean_keywords(response.key_phrases)
            return keywords
        except Exception as e:
            print(f"Encountered exception. {e}")

    def _clean_keywords(self, keywords: List[str]) -> List[str]:
        return [kw for kw in keywords if kw not in self.NON_KEYWORDS]
