from add_entry import change_style
import dotenv
import json

def create_dataset(input_file, output_file):
    """
    Creates a dataset by transforming text styles using the change_style function.

    Args:
        input_file (str): Path to the input file containing text entries.
        style (str): The target style for transformation.
        output_file (str): Path to save the transformed dataset.
    """
    dotenv.load_dotenv()
    with open(input_file, 'r', encoding='utf-8') as f:
        inputs = f.readlines()

    style = "Non-toxic"  # The target style needs to be gotten from the input file

    # Iterate through each entry in the input file
    transformed_entries = []
    transformed_entries.append("timestamp,input,style,prompt,response,model\n")
    k = 0
    k_transformed_entries = []
    for entry in inputs:
        transformed_entry = change_style(entry.strip(), style)
        if transformed_entry is not None:
            k_transformed_entries.append(
                f"{transformed_entry['timestamp']},{transformed_entry['input']},{transformed_entry['style']},\"{transformed_entry['prompt']}\",\"{transformed_entry['response']}\",{transformed_entry['model']}\n")
            k += 1

        if k % 10 == 0 and k > 0:
            transformed_entries.extend(k_transformed_entries)
            k_transformed_entries = []
            with open(output_file, 'a', encoding='utf-8') as f:
                json.dumps(transformed_entries)


