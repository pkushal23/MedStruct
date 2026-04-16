import spacy
import sys

def check_dependencies():
    print("Checking UMLS linker dependencies...")
    try:
        if spacy.util.is_package("en_core_sci_sm"):
            print("Dependencies verified: 'en_core_sci_sm' is installed.")
            return True
        else:
            raise ImportError
    except Exception:
        print("\n" + "="*60)
        print("ERROR: SciSpacy medical model is missing!")
        print("The UMLS linker requires the 'en_core_sci_sm' model weights.")
        print("Please run the following command in your terminal:")
        print("pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz")
        print("="*60 + "\n")
        return False