import glob
import LlmBase
from utils_extraction import read_txt_file_as_string, load_files


class LlmExtraction(LlmBase.GptLLM):

    def __init__(self, model_name, temperature=0.5):
        super().__init__(model_name, temperature)
        self.prompt_path = "extract_information_prompt.txt"
        self.system_input = """As an expert in ontology engineering, I would like your support in extracting detailed information for a set of specified variables from the provided ontology."""
        self.prompt = ""

    def set_prompt(self, variables, ontology_content):
        initial_prompt = read_txt_file_as_string(self.prompt_path)
        initial_prompt = initial_prompt.format(variables=variables,ontology_content=ontology_content)
        conversation = []
        conversation.append({"role": "system", "content": self.system_input})
        conversation.append({"role": "user", "content": initial_prompt})
        self.prompt = conversation

    def get_prompt(self):
        return self.prompt

    def clean_responses(self):
        super().clean_responses()


def main(model_name,variables,ontology):

    with open(ontology, "r", encoding="utf-8") as file:
        ontology_content = file.read() #get the content of the ontology


    llm_extraction = LlmExtraction(model_name)

    llm_extraction.set_prompt(variables,ontology_content) #include variables and ontology into the prompt

    llm_extraction.run_inference(llm_extraction.prompt) #run the query

    llm_extraction.save_responses_to_json("./information_extracted.txt")
    llm_extraction.clean_responses()


if __name__ == "__main__":
    model_name = "gpt-4o"
    variables = ["alcohol consumption", "bmi", "comorbidity index", "education", "fear avoidance", "financial strain", "pain catastrophizing score", "peg score", "PROMIS anxiety", "PROMIS depression", "sleep disturbance"]
    ontology = "backpain_ontology.ttl"
    main(model_name,variables,ontology)
