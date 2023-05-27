from datasets import load_dataset

def split_input(example):
    _input = example["input"]
    prompt = "<bot>: ".join(_input.split("<bot>: ")[:-1]) + "<bot>: "
    answer = _input.split("<bot>: ")[-1]
    answer_without_human = answer.split("<human>: ")[0]
    return {"input": prompt, "output": answer_without_human}

def replace_h2ogpt(example):
    example["input"] = example["input"].replace("h2oGPT", "Eagle")
    example["input"] = example["input"].replace("H2O.ai", "Björn")
    return example

def prepare_oasst_data():
    dataset = load_dataset("h2oai/openassistant_oasst1_h2ogpt", cache_dir="./datasets_cache")
    # Dataset has only input column but needs input + output columns
    # so we split the input before the last <bot> answer
    dataset = dataset.map(replace_h2ogpt)
    dataset = dataset.map(split_input, remove_columns=["prompt_type", "source"])
    dataset.push_to_hub("bjoernp/oasst1_processed")

if __name__ == "__main__":
    prepare_oasst_data()