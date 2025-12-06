import json


def inspect_euc2d(path):
    with open(path, "r") as f:
        data = json.load(f)

    print(f"Total de instâncias no arquivo {path}: {len(data)}")
    for idx, entry in enumerate(data):
        num_cities = len(entry["coords"])
        # print(f"Instância {idx}: {num_cities} cidades")

    # quantas instancias tem em cada quantidade de cidades
    city_count = {}
    for entry in data:
        n = len(entry["coords"])
        if n not in city_count:
            city_count[n] = 0
        city_count[n] += 1

    print("Quantidade de instâncias por número de cidades:")
    for num_cities, count in sorted(city_count.items()):
        print(f"{num_cities} cidades: {count} instâncias")


if __name__ == "__main__":
    inspect_euc2d("data/EUC_2D.json")
