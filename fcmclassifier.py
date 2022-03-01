from genericpath import exists
import click
import os
from classification.classify_from_nrcs import classify as classify
from nrc.nrc_calc import main as nrc_calc
from quantization.quantize_folder import main as quantize_folder


@click.command()
@click.argument('input_dir', nargs=1, type=click.Path(exists=True))
@click.argument('output_dir', nargs=1, type=click.Path())
@click.option('--alphabet', '-a', type=int, help="Alphabet size used in quantization")
@click.option('--k', '-k', type=int, help="Context size to be used for FCM")
@click.option('--d', '-d', type=int, default=1, help="Depth size to be used for FCM")
@click.option('--threads', '-t', default=1, type=int, help="Maximum number of threads used")
@click.option('--average_over', '-v', type=int, default=1, help="Downsampling level (average over)")
@click.option('--results_nrc', '-o', default="results_nrc.csv", type=str, help="Output file with the NRC values")
def main(input_dir, output_dir, alphabet, k, d, threads, average_over, results_nrc):

    print("\n\n--------------------------------")
    print("        QUANTIZE_FOLDER         ")
    print("--------------------------------\n")
    os.system("rm -rf {0}*".format(output_dir))
    quantize_folder(input_dir=input_dir, output_dir=output_dir, 
                    alphabet=alphabet, average_over=average_over)

    print("\n\n--------------------------------")
    print("            NRC_CALC            ")
    print("--------------------------------\n")
    nrc_calc(input_dir=output_dir, alphabet=alphabet, k=k, d=d, output_csv=results_nrc,
                       threads=threads)
    
    print("\n\n--------------------------------")
    print("           CLASSIFY             ")
    print("--------------------------------\n")
    classify(results_nrc)


if __name__ == "__main__":
    main()