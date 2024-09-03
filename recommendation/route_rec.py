import re
from itertools import permutations

def parse_documents(documents):
    """Parse the documents to extract latitude and longitude."""
    coords = []
    for doc in documents:
        latitude = re.search(r'latitude:\s*([\d.]+)', doc.page_content).group(1)
        longitude = re.search(r'longitude:\s*([\d.]+)', doc.page_content).group(1)
        coords.append((float(latitude), float(longitude)))
    return coords


def format_docs_with_coords(docs):
    coords = parse_documents(docs)
    context = format_docs(docs)
    print('context:', context,'coords:', coords)
    return coords, context

def find_shortest_route(coords):
    """Find the route with the shortest total Manhattan distance."""
    
    def manhattan_distance(coord1, coord2):
        """Calculate the Manhattan distance between two coordinates."""
        return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])

    def total_manhattan_distance(route, coords):
        """Calculate the total Manhattan distance for a given route."""
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += manhattan_distance(coords[route[i]], coords[route[i + 1]])
        print(total_distance)
        return total_distance

    num_locations = len(coords)
    all_permutations = permutations(range(num_locations))
    # print(list(all_permutations))
    min_distance = float('inf')
    best_route = None
    
    for perm in all_permutations:
        current_distance = total_manhattan_distance(perm, coords)
        if current_distance < min_distance:
            min_distance = current_distance
            best_route = perm
    
    return best_route, min_distance

def rec_rag_chain_logic(retriever, query):
    docs = retriever.invoke(query)
    coords, context = format_docs_with_coords(docs)
    best_route, min_distance = find_shortest_route(coords)
    
    formatted_route = ", ".join([docs[i].page_content.split('\n')[0].split(': ')[1] for i in best_route])
    answer = f"최적의 동선은 다음과 같습니다: {formatted_route}. 총 맨하탄 거리는 {min_distance}입니다."
    
    return answer
