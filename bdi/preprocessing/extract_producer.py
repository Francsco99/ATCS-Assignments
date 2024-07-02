import os
import json

monitor_manufacturers = ["3m","apc","etronix","princeton digital","qnix",
    "aoc", "aopen", "acer", "acer predator", "advantech", "ag neovo", "alienware", 
    "apple", "asus", "asus rog","avocent" "barco", "benq", "belinea", "belkin", "boe", 
    "braun", "chimei innolux", "changhong", "ctx", "coby", "cornea", "corsair", 
    "curtis", "daewoo", "dell", "double sight","doublesight", "edge10","eaton", "e-machines", "eizo", "elo touch",
    "elo touch solutions","elo", "element", "enermax", "envizex", "ergotron","faytech", "fujitsu", 
    "funai", "gigabyte", "goldstar", "grundig", "gvision", "hannspree", "hanns.g", 
    "handheld", "handheld us","hannsg", "hercules","hewlett", "hewlett-packard","hewlett packard", "hitachi", "hp", 
    "iiyama", "insignia", "intey", "jvc", "key technology corporation", "konka", 
    "ktc", "lacie", "lenovo", "lenovo legion", "lg", "lg ultrawide", "littlebigdisk", 
    "loewe", "logisys", "magnavox", "mag innovision", "mag", "medion", "metz", 
    "mimo", "mitsubishi", "msi", "nanovision", "nds surgical imaging", "nec", 
    "neovo","newstar", "nixeus", "optoma", "packard bell", "panasonic", "philips", "pixo", 
    "planar","planar systems", "polaroid", "polytron", "primax", "proscan", "proton", 
    "protron", "proview", "quasar", "razer", "realistic", "runco", "sampo", 
    "samsung", "sanyo", "sceptre", "seiki", "sharp","sunbrite", "shuttle", "siemens", 
    "singer", "skyworth", "sony", "spec research", "startech","supersonic", "tatung", 
    "telefunken", "tenaus", "totoku", "toshiba", "v7", "verbatim", "vestel", 
    "viewpia", "viewsonic", "vistaquest", "vivitek", "vizio", "wacom", 
    "westinghouse","wortmann ag", "xiaomi", "zowie", "zyxel","packard"
]

relevant_labels = ["brand","brand name","manufacturer","publisher"]

manufacturer_mapping = {
    "hewlett-packard": "hp",
    "hewlett packard": "hp",
    "hewlett":"hp",
    "packard":"hp",
    "elo touch solutions": "elo",
    "elo touch": "elo",
    "lenovo legion": "lenovo",
    "double sight": "doublesight"
}

def extract_producer(data):
    for label in relevant_labels:
        if label in data:
            if isinstance(data[label], list):
                value = data[label][0]
            else:
                value = data[label]

            # Gestisci le varianti del produttore usando la mappa
            if value.lower() in manufacturer_mapping:
                return manufacturer_mapping[value.lower()]
            
            # Estrai la prima parola e mettila in lowercase
            first_word = value.split()[0].lower()
            return first_word
    
    # Se nessuna delle etichette rilevanti è trovata, usa '<page title>'
    if '<page title>' in data:
        page_title = data['<page title>'].lower()
        # Cerca una corrispondenza con i produttori nella lista
        for manufacturer in monitor_manufacturers:
            if manufacturer in page_title:
                if manufacturer.lower() in manufacturer_mapping:
                    return manufacturer_mapping[manufacturer.lower()]
                return manufacturer

    return None
    
def process_sources_producers(root_path,sources_path):
    item_to_producer={}
    for k, v, files in os.walk(root_path + '/' + sources_path):
        for file in files:
            file_path = os.path.join(root_path+'/'+sources_path,file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data is not None:
                        item = sources_path + '//' + file.replace('.json','')
                        producer = extract_producer(data)
                        item_to_producer[item]=producer
                    else:
                        print(f"File json non esistente o vuoto: {file_path}")
            except Exception as e:
                print(f"Errore nella apertura o lettura di {file_path}: {e}")

    return item_to_producer

def process_sources_producers_main(root_path):
    item_to_producer = {}

    for k, dirs, v in os.walk(root_path):
        for dir in dirs:
            item_to_producer.update(process_sources_producers(root_path, dir))

    try:
        proc_path = os.path.join('preprocessing', 'item_to_producer.json')
        with open(proc_path, 'w', encoding='utf-8') as f:
            json.dump(item_to_producer, f, ensure_ascii=False, indent=4)
        print(f"Written to {proc_path}")
    except Exception as e:
        print(f"Error writing {proc_path}: {e}")
    
process_sources_producers_main('monitor/monitor_specs')