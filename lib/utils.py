import torch


COUNTRY_MAP = {
    "at": "Austria",
    "ba": "Bosnia and Herzegovina",
    "be": "Belgium",
    "bg": "Bulgaria",
    "cz": "Czechia",
    "dk": "Denmark",
    "ee": "Estonia",
    "es": "Spain",
    "es-ct": "Catalonia",
    "es-ga": "Galicia",
    "es-pv": "Basque Country",
    "fi": "Finland",
    "fr": "France",
    "gb": "Great Britain",
    "gr": "Greece",
    "hr": "Croatia",
    "hu": "Hungary",
    "is": "Iceland",
    "it": "Italy",
    "lv": "Latvia",
    "nl": "The Netherlands",
    "no": "Norway",
    "pl": "Poland",
    "pt": "Portugal",
    "rs": "Serbia",
    "se": "Sweden",
    "si": "Slovenia",
    "tr": "Turkey",
    "ua": "Ukraine"
}


def check_cuda_memory():
    """Convenience function to check CUDA memory usage
    """
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(i))
        
    total = torch.cuda.get_device_properties(0).total_memory
    reserved = torch.cuda.memory_reserved(0)
    allocated = torch.cuda.memory_allocated(0)
    free = reserved - allocated

    print(f"Total: {total / 1e6:,.2f} MB")
    print(f"Reserved: {reserved / 1e6:,.2f} MB")
    print(f"Allocated: {allocated / 1e6:,.2f} MB")
    print(f"Free in reserved: {free / 1e6:,.2f} MB")
