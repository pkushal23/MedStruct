import spacy
import sys

def check_dependencies():
    """Verify that the required SciSpacy models and linkers are available."""
    print("Checking UMLS linker dependencies...")
    try:
        # Prefer en_core_sci_lg (600k vectors) for semantic resolution
        if spacy.util.is_package("en_core_sci_lg"):
            print("Dependencies verified: 'en_core_sci_lg' (large model) is installed.")
            return True
        elif spacy.util.is_package("en_core_sci_sm"):
            print("WARNING: Using 'en_core_sci_sm' (small model).")
            print("For better synonym resolution, run: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz")
            return True
        else:
            print("\nERROR: SciSpacy medical model missing. Please install en_core_sci_lg or en_core_sci_sm.")
            return False
    except Exception as e:
        print(f"Dependency check failed: {e}")
        return False