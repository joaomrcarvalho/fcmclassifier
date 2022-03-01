from pyexpat import model
from quantization.diffquantizer import DiffQuantizer
from pathlib import Path
import numpy as np
from tqdm import tqdm
from utils.utils import if_dir_does_not_exist_create


def get_all_files(dirpath):
    file_list = []
    for x in dirpath.iterdir():
        if x.is_file():
            file_list.append(x)
        elif x.is_dir():
            file_list.extend(get_all_files(x))
    return file_list


def main(input_dir, output_dir, alphabet, average_over=1):
    """
        Quantizer for 1D signals using first-order derivatives and optional filtering.
    """
    input_dir = Path(input_dir)
    assert input_dir.is_dir()

    output_dir = Path(output_dir)
    if_dir_does_not_exist_create(output_dir)
    assert output_dir.is_dir()

    input_training_dir = input_dir / "train"
    input_test_dir = input_dir / "test"

    output_training_dir = output_dir / "train"
    if_dir_does_not_exist_create(output_training_dir)
    output_test_dir = output_dir / "test"
    if_dir_does_not_exist_create(output_test_dir)

    quantizer = DiffQuantizer(alphabet_size=alphabet, average_over=average_over)

    for curr_training_dir in input_training_dir.iterdir():
        print("\nProcessing folder", curr_training_dir)

        class_name = curr_training_dir.name
        content = np.empty(shape=(0))

        for file in tqdm(get_all_files(curr_training_dir)):
            tmp_content = quantizer.read_csv_file(file)
            content = np.concatenate((content, tmp_content))

        print("Learning breakpoints for ", class_name)
        breakpoints = quantizer.learn_breakpoints(content)
        quantized_output = quantizer.perform_quantization(content, breakpoints=breakpoints)
        curr_output_folder = Path(output_training_dir / class_name)
        if not curr_output_folder.exists():
            curr_output_folder.mkdir()

        np.savetxt("{0}".format(Path(curr_output_folder / class_name)), quantized_output, fmt='%s', newline='')


        model_being_used = class_name
        model_output_folder = Path(output_test_dir / model_being_used)
        if_dir_does_not_exist_create(model_output_folder)
        test_file_counter = 0

        print("\n\nCurrent model being used for quantization is", model_being_used)
        for curr_test_dir in input_test_dir.iterdir():
            print("\nProcessing folder", curr_test_dir)

            test_class_name = curr_test_dir.name
            curr_output_folder = Path(model_output_folder / test_class_name)
            if_dir_does_not_exist_create(curr_output_folder)

            files = get_all_files(curr_test_dir)
            for i in tqdm(range(len(files))):
                file = files[i]
                content = quantizer.read_csv_file(file)

                quantized_output = quantizer.perform_quantization(content, breakpoints=breakpoints)
                test_file_counter += 1

                np.savetxt("{0}".format(Path(curr_output_folder / str(test_file_counter))),
                           quantized_output, fmt='%s', newline='')


if __name__ == "__main__":
    main()



