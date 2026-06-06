import os

test_paths = {
    "2channel1dcnn": "/Users/aakashrajput/MachineLearning/Exoplanets/src/2channel1dcnn/test.py",
    "original1dcnn": "/Users/aakashrajput/MachineLearning/Exoplanets/src/original1dcnn/test.py",
    "optimized1dcnn": "/Users/aakashrajput/MachineLearning/Exoplanets/src/optimized1dcnn/test.py",
    "transformerarch": "/Users/aakashrajput/MachineLearning/Exoplanets/src/transformerarch/test.py"
}

for name, path in test_paths.items():
    if not os.path.exists(path):
        print(f"ERROR: {path} not found")
        continue
    
    with open(path, "r") as f:
        code = f.read()
        
    # 1. Inject torchinfo import at the top
    if "import torchinfo" not in code:
        code = "import torchinfo\n" + code
        
    # 2. Update eval_size from 1000 to 5000
    code = code.replace("eval_size = min(1000, len(val_dataset))", "eval_size = min(5000, len(val_dataset))")
    
    # 3. Insert the model summary calculation after the model instantiation
    model_cnn_line = "model = NasaInaraModel(in_channels=in_channels, sequence_length=seq_len).to(device)"
    model_trans_line = "model = NasaInaraTransformer(in_channels=in_channels, sequence_length=seq_len).to(device)"
    
    summary_injection = """
    # Generate torchinfo summary string
    try:
        model_summary_str = str(torchinfo.summary(model, input_size=(1, in_channels, seq_len), device="cpu", verbose=0))
    except Exception as e:
        model_summary_str = f"Error generating model summary: {e}"
    """
    
    if model_cnn_line in code and "model_summary_str =" not in code:
        code = code.replace(model_cnn_line, model_cnn_line + summary_injection)
    elif model_trans_line in code and "model_summary_str =" not in code:
        code = code.replace(model_trans_line, model_trans_line + summary_injection)
        
    # 4. Append model summary to report_content before saving to details.txt
    write_line = 'with open(details_path_1, "w") as f:'
    append_block = """report_content += f"\\n\\nMODEL ARCHITECTURE SUMMARY:\\n{'-'*80}\\n{model_summary_str}\\n{'-'*80}\\n"
    with open(details_path_1, "w") as f:"""
    
    if write_line in code and "MODEL ARCHITECTURE SUMMARY:" not in code:
        code = code.replace(write_line, append_block)
        
    # 5. Fix hardcoded path replacements if any
    if name == "optimized1dcnn":
        code = code.replace("10_mil_minus_params", "optimized1dcnn")
    elif name == "transformerarch":
        code = code.replace("10_mil_minus_params/transformer", "transformerarch")
        code = code.replace("10_mil_minus_params.transformer", "transformerarch")
        
    with open(path, "w") as f:
        f.write(code)
    print(f"Successfully updated {name}/test.py")
