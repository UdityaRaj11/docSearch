from flask import Flask, request, jsonify
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

app = Flask(__name__)
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

class Diagnosis(BaseModel):
    type: list[str] = Field(..., description="Types of doctor to visit")

class DiagnosisResponse(BaseModel):
    diagnoses: list[Diagnosis]

parser = PydanticOutputParser(pydantic_object=DiagnosisResponse)

prompt = PromptTemplate(
    template=(
        "Based on the symptom, age and gender '{symptom}', provide "
        "the corresponding types of doctors to visit. Return JSON strictly in the format: \n\n"
        "{format_instructions}\n\n"
        "No additional text or explanations."
    ),
    input_variables=["symptom"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

def get_diagnosis(symptom: str) -> dict:
    """Generates a structured diagnosis response using the Gemini model."""
    formatted_prompt = prompt.format(symptom=symptom)
    response = model.invoke(formatted_prompt)

    try:
        result = parser.parse(response.content)
        return result.model_dump()
    except Exception as e:
        return {"error": f"Parsing error: {str(e)}"}

@app.route("/diagnosis", methods=["POST"])
def diagnosis():
    """API endpoint to get a diagnosis based on input symptoms."""
    data = request.get_json()
    symptom = data.get("symptom")
    
    if not symptom:
        return jsonify({"error": "Missing symptom parameter"}), 400
    
    diagnosis_result = get_diagnosis(symptom)
    return jsonify(diagnosis_result)

if __name__ == "__main__":
    app.run(debug=True)
