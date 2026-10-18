import fitz

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class ReadPDFInput(BaseModel):
    paper_path: str = Field(
        ...,
        description="Absolute path to the research paper PDF."
    )


class ReadPDFTool(BaseTool):
    name: str = "Read Research Paper PDF"

    description: str = (
        "Reads a research paper PDF and extracts its text. "
        "Return the most relevant research content while keeping "
        "the output compact enough for downstream analysis."
    )

    args_schema: type[BaseModel] = ReadPDFInput

    def _run(self, paper_path: str) -> str:

        try:
            document = fitz.open(paper_path)

            pages = []

            for page_number, page in enumerate(document, start=1):

                text = page.get_text()

                pages.append(
                    f"\n===== PAGE {page_number} =====\n{text}"
                )

            document.close()

            full_text = "\n".join(pages)

            # Keep the proof-of-concept request below Groq's TPM limit.
            # Roughly 4 characters ~= 1 token.
            max_chars = 28000

            if len(full_text) > max_chars:
                full_text = (
                    full_text[:max_chars]
                    + "\n\n[PDF TEXT TRUNCATED FOR API TOKEN LIMIT]\n"
                )

            return full_text

        except Exception as e:

            return f"ERROR: Could not read PDF: {e}"


read_pdf_tool = ReadPDFTool()