import random
import math

import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag

import xml.etree.ElementTree as ET

# NLTK DOWNLOADS
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


# PARAMETERS FROM PAPER
EA = 16         # energy taken by an ant when it arrives on a node
EMAX = 56       # maximum quantity of energy an ant can carry
DELTA = 0.9     # evaporation rate of the pheromone between 2 cycles
E0 = 30         # initial quantity of energy on each node
OMEGA = 25      # ant life span
LV = 100        # odour vector length
DELTA_V = 0.9   # percentage of the odour vector components deposited by an ant when it arrives on a node
C_AC = 100      # number of cycles of the simulation
THETA = 1.0     # quantity of deposited pheromone


def load_semeval_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    texts = []
    for text in root.iter("text"):
        text_sentences = []
        for sentence in text.iter("sentence"):
            sentence_words = []
            for element in sentence:
                word = element.text
                if word is None:
                    continue
                sentence_words.append({
                    "word": word.lower(),
                    "id": element.attrib.get("id"),
                    "lemma": element.attrib.get("lemma"),
                    "pos": element.attrib.get("pos")
                })
            if sentence_words:
                text_sentences.append(sentence_words)
        if text_sentences:
            texts.append(text_sentences)
    return texts

def load_gold_key(key_path):
    gold = {}
    with open(key_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            instance_id = parts[1]
            sense_keys = []
            for p in parts[2:]:
                if p == "!!":
                    break
                sense_keys.append(p)
            if sense_keys:
                gold[instance_id] = sense_keys
    return gold


# CLASSES
class Node:
    def __init__(self, node_id, node_type, label=None):
        self.id = node_id
        self.type = node_type
        self.label = label
        self.energy = E0
        self.odour_vector = []
        self.neighbors = []

    def add_neighbor(self, node):
        if node not in self.neighbors:
            self.neighbors.append(node)

    def add_energy(self, quantity):
        self.energy = self.energy + quantity

    def remove_energy(self, quantity):
        taken = min(self.energy, quantity)
        self.energy = self.energy - taken
        return taken

    def deposit_odour(self, odour_components):
        for component in odour_components:
            if len(self.odour_vector) < LV:
                self.odour_vector.append(component)
            else:
                random_index = random.randint(0, LV - 1)
                self.odour_vector[random_index] = component


class NestNode(Node):
    def __init__(self, node_id, synset, word, instance_id=None):
        super().__init__(node_id, "nest", label=synset.name())
        self.synset = synset
        self.word = word
        self.instance_id = instance_id
        self.definition = synset.definition()
        self.odour_vector = []
        self.ants = []


class Edge:
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
        self.pheromone = 0.0
        self.is_bridge = False

    def add_pheromone(self, quantity):
        self.pheromone = self.pheromone + quantity

    def evaporate(self, delta):
        self.pheromone *= (1 - delta)
        if self.pheromone < 0:
            self.pheromone = 0


class Bridge(Edge):
    def __init__(self, node1, node2):
        super().__init__(node1, node2)
        self.is_bridge = True

    def should_collapse(self):
        return self.pheromone <= 0


class Ant:
    def __init__(self, ant_id, mother_nest, lifespan):
        self.id = ant_id
        self.mother_nest = mother_nest
        self.current_node = mother_nest
        self.previous_node = None
        self.energy = 0
        self.lifespan = lifespan
        self.mode = "explore" # "explore" or "return"
        self.odour_vector = mother_nest.odour_vector.copy()
        self.path = []
        self.alive = True

    def decrease_lifespan(self):
        self.lifespan = self.lifespan - 1
        if self.lifespan <= 0:
            self.alive = False

    def switch_to_return_mode(self):
        self.mode = "return"

    def move_to(self, next_node):
        self.previous_node = self.current_node
        self.current_node = next_node
        self.path.append(next_node)


# EXTENDED LESK

word_to_id = {}
next_word_id = 0

def get_word_id(word):
    global next_word_id
    word = word.lower()
    if word not in word_to_id:
        word_to_id[word] = next_word_id
        next_word_id = next_word_id + 1
    return word_to_id[word]

def tokenize_definition(text):
    return [token.lower() for token in word_tokenize(text) if token.isalpha()]

def get_extended_definition_words(synset):
    words = []
    words.extend(tokenize_definition(synset.definition()))
    for example in synset.examples():
        words.extend(tokenize_definition(example))
    related_synsets = (
        synset.hypernyms() + synset.hyponyms() +
        synset.member_holonyms() + synset.part_holonyms() +
        synset.substance_holonyms() + synset.member_meronyms() +
        synset.part_meronyms() + synset.substance_meronyms()
    )
    for related in related_synsets:
        words.extend(tokenize_definition(related.definition()))
    return words

def build_odour_vector(synset):
    words = get_extended_definition_words(synset)
    vector = sorted({get_word_id(word) for word in words})
    return vector[:LV]

_lesk_nest_cache = {}

def ext_lesk(vector1, vector2):
    return len(set(vector1).intersection(set(vector2)))

def ext_lesk_nests(nest_vector1, nest_vector2):
    key = (id(nest_vector1), id(nest_vector2)) if id(nest_vector1) <= id(nest_vector2) \
        else (id(nest_vector2), id(nest_vector1))
    cached = _lesk_nest_cache.get(key)
    if cached is not None:
        return cached
    result = len(set(nest_vector1).intersection(set(nest_vector2)))
    _lesk_nest_cache[key] = result
    return result


# PREPROCESSING

def penn_to_wordnet_pos(tag):
    if tag.startswith("N"): return wn.NOUN
    if tag.startswith("V"): return wn.VERB
    if tag.startswith("J"): return wn.ADJ
    if tag.startswith("R"): return wn.ADV
    return None

def semeval_pos_to_wordnet_pos(pos):
    if pos is None: return None
    pos = pos.lower()
    if pos.startswith("n"): return wn.NOUN
    if pos.startswith("v"): return wn.VERB
    if pos.startswith("j") or pos.startswith("a"): return wn.ADJ
    if pos.startswith("r"): return wn.ADV
    return None

def preprocess_semeval_text(semeval_text):
    processed_sentences = []
    for sentence in semeval_text:
        processed_sentence = []
        for item in sentence:
            word = item["lemma"] if item["lemma"] else item["word"]
            wn_pos = semeval_pos_to_wordnet_pos(item["pos"])
            if wn_pos is None:
                continue
            synsets = wn.synsets(word.lower(), pos=wn_pos)
            if not synsets:
                continue
            processed_sentence.append({
                "word": word.lower(),
                "id": item["id"],
                "pos": wn_pos,
                "synsets": synsets
            })
        if processed_sentence:
            processed_sentences.append(processed_sentence)
    return processed_sentences

def preprocess_text(text):
    processed_sentences = []
    for sentence in sent_tokenize(text):
        tokens = word_tokenize(sentence)
        tagged_tokens = pos_tag(tokens)
        processed_sentence = []
        for word, tag in tagged_tokens:
            if not word.isalpha():
                continue
            wn_pos = penn_to_wordnet_pos(tag)
            if wn_pos is None:
                continue
            synsets = wn.synsets(word.lower(), pos=wn_pos)
            if not synsets:
                continue
            processed_sentence.append({
                "word": word.lower(),
                "id": None,
                "pos": wn_pos,
                "synsets": synsets
            })
        if processed_sentence:
            processed_sentences.append(processed_sentence)
    return processed_sentences


# GRAPH BUILDING

class GraphBuilder:
    def __init__(self):
        self.nodes = {}
        self.edges = []

        self.edge_map = {}
        self.node_counter = 0
        self.word_to_nests = {}

    def create_node(self, node_type, label=None):
        node = Node(self.node_counter, node_type, label)
        self.nodes[self.node_counter] = node
        self.node_counter = self.node_counter +1
        return node

    def create_nest(self, synset, word, instance_id=None):
        nest = NestNode(self.node_counter, synset, word, instance_id)
        nest.odour_vector = build_odour_vector(synset)
        self.nodes[self.node_counter] = nest
        self.node_counter = self.node_counter + 1
        return nest

    def connect(self, node1, node2):
        edge = Edge(node1, node2)
        self.edges.append(edge)
        node1.add_neighbor(node2)
        node2.add_neighbor(node1)

        key = (min(node1.id, node2.id), max(node1.id, node2.id))
        self.edge_map[key] = edge
        return edge

    def build_graph(self, processed_sentences):
        text_node = self.create_node("text", "TEXT")
        for sentence_index, sentence in enumerate(processed_sentences):
            sentence_node = self.create_node("sentence", f"sentence_{sentence_index}")
            self.connect(text_node, sentence_node)
            for word_data in sentence:
                word = word_data["word"]
                instance_id = word_data.get("id")
                synsets = word_data["synsets"]
                word_node = self.create_node("word", word)
                self.connect(sentence_node, word_node)
                self.word_to_nests[word_node.id] = {
                    "word": word,
                    "instance_id": instance_id,
                    "word_node": word_node,
                    "nests": []
                }
                for synset in synsets:
                    nest = self.create_nest(synset, word, instance_id)
                    self.connect(word_node, nest)
                    self.word_to_nests[word_node.id]["nests"].append(nest)
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "edge_map": self.edge_map,
            "text_node": text_node,
            "word_to_nests": self.word_to_nests
        }


# ANT COLONY ALGORITHM

class AntColony:
    def __init__(self, graph_data):
        self.nodes = graph_data["nodes"]
        self.edges = graph_data["edges"]

        self.edge_map = graph_data["edge_map"]
        self.word_to_nests = graph_data["word_to_nests"]

        self.ants = []
        self.ant_counter = 0

        self.best_configuration = None
        self.best_score = -1

    def get_edge(self, node1, node2):
        key = (min(node1.id, node2.id), max(node1.id, node2.id))
        return self.edge_map.get(key, None)

    def create_bridge(self, nest1, nest2):
        existing = self.get_edge(nest1, nest2)
        if existing is not None:
            return existing
        bridge = Bridge(nest1, nest2)
        bridge.add_pheromone(THETA)
        self.edges.append(bridge)
        nest1.add_neighbor(nest2)
        nest2.add_neighbor(nest1)
        key = (min(nest1.id, nest2.id), max(nest1.id, nest2.id))
        self.edge_map[key] = bridge
        return bridge

    def remove_collapsed_bridges(self):
        remaining_edges = []
        for edge in self.edges:
            if edge.is_bridge and edge.should_collapse():
                key = (min(edge.node1.id, edge.node2.id), max(edge.node1.id, edge.node2.id))
                self.edge_map.pop(key, None)
                if edge.node2 in edge.node1.neighbors:
                    edge.node1.neighbors.remove(edge.node2)
                if edge.node1 in edge.node2.neighbors:
                    edge.node2.neighbors.remove(edge.node1)
            else:
                remaining_edges.append(edge)
        self.edges = remaining_edges

    def produce_ant_probability(self, nest):
        return math.atan(nest.energy) / math.pi + 0.5

    def produce_ants(self):
        for word_data in self.word_to_nests.values():
            for nest in word_data["nests"]:
                probability = self.produce_ant_probability(nest)
                if random.random() < probability and nest.energy >= 1:
                    nest.energy = nest.energy - 1
                    ant = Ant(
                        ant_id=self.ant_counter,
                        mother_nest=nest,
                        lifespan=OMEGA
                    )
                    self.ant_counter = self.ant_counter + 1
                    self.ants.append(ant)
                    nest.ants.append(ant)

    def remove_dead_ants(self):
        alive_ants = []
        for ant in self.ants:
            if ant.alive:
                alive_ants.append(ant)
            else:
                ant.current_node.add_energy(ant.energy)
        self.ants = alive_ants

    def collect_energy(self, ant):
        taken = ant.current_node.remove_energy(EA)
        ant.energy = ant.energy + taken
        if ant.energy > EMAX:
            extra = ant.energy - EMAX
            ant.energy = EMAX
            ant.current_node.add_energy(extra)

    def should_return_home(self, ant):
        probability = ant.energy / EMAX
        return random.random() < probability

    def evaluate_explore_transition(self, node, edge, neighbors):
        total_energy = sum(neighbor.energy for neighbor in neighbors) + 1e-9
        node_score = node.energy / total_energy
        edge_score = max(0, 1 - edge.pheromone)
        return node_score + edge_score

    def _lesk_with_mother(self, node, mother_nest):
        if node.type == "nest":
            return ext_lesk_nests(node.odour_vector, mother_nest.odour_vector)
        return ext_lesk(node.odour_vector, mother_nest.odour_vector)

    def evaluate_return_transition(self, edge, similarity, total_similarity):
        node_score = similarity / total_similarity
        edge_score = edge.pheromone
        return node_score + edge_score

    def choose_next_node(self, ant):
        neighbors = [n for n in ant.current_node.neighbors if n != ant.previous_node]
        if not neighbors:
            neighbors = ant.current_node.neighbors
        if not neighbors:
            return None

        candidates = []

        if ant.mode == "return":
            similarities = [self._lesk_with_mother(n, ant.mother_nest) for n in neighbors]
            total_similarity = sum(similarities) + 1e-9

        for i, neighbor in enumerate(neighbors):
            edge = self.get_edge(ant.current_node, neighbor)
            if edge is None:
                continue
            if ant.mode == "explore":
                score = self.evaluate_explore_transition(neighbor, edge, neighbors)
            else:
                score = self.evaluate_return_transition(
                    edge,
                    similarities[i],
                    total_similarity
                )
            candidates.append((neighbor, score))

        if not candidates:
            return None

        total_score = 0

        for node, score in candidates:
            total_score = total_score + score

        if total_score <= 0:
            return random.choice([node for node, score in candidates])

        nodes = []
        probabilities = []

        for node, score in candidates:
            probability = score / total_score

            nodes.append(node)
            probabilities.append(probability)

        chosen_node = random.choices(nodes, weights=probabilities, k=1)[0]

        return chosen_node

    def move_ant(self, ant):
        if ant.mode == "explore":
            self.collect_energy(ant)
            if self.should_return_home(ant):
                ant.switch_to_return_mode()

        next_node = self.choose_next_node(ant)
        if next_node is not None:
            edge = self.get_edge(ant.current_node, next_node)
            if edge is not None:
                edge.add_pheromone(THETA)
            ant.move_to(next_node)

        ant.decrease_lifespan()

    def move_ants(self):
        for ant in self.ants:
            self.move_ant(ant)

    def deposit_odour_on_node(self, ant):
        if ant.current_node.type == "nest":
            return
        if not ant.odour_vector:
            return
        number_to_deposit = max(1, int(len(ant.odour_vector) * DELTA_V))
        components = random.sample(
            ant.odour_vector,
            min(number_to_deposit, len(ant.odour_vector))
        )
        ant.current_node.deposit_odour(components)

    def is_mother_nest(self, ant, node):
        return node.type == "nest" and node == ant.mother_nest

    def is_enemy_nest(self, ant, node):
        return (
            node.type == "nest"
            and node != ant.mother_nest
            and node.word == ant.mother_nest.word
        )

    def is_potential_friend_nest(self, ant, node):
        return (
            node.type == "nest"
            and node != ant.mother_nest
            and node.word != ant.mother_nest.word
        )

    def is_plain_node(self, node):
        return node.type != "nest"

    def try_create_bridge(self, ant):
        current = ant.current_node
        for neighbor in current.neighbors:
            if self.is_potential_friend_nest(ant, neighbor):
                score = ext_lesk_nests(ant.mother_nest.odour_vector, neighbor.odour_vector)
                if score > 0:
                    self.create_bridge(ant.mother_nest, neighbor)
                    ant.switch_to_return_mode()
                    return

    def evaporate_pheromones(self):
        for edge in self.edges:
            edge.evaporate(DELTA)

    def get_current_configuration(self):
        configuration = {}
        for word_node_id, word_data in self.word_to_nests.items():
            nests = word_data["nests"]
            best_nest = max(nests, key=lambda nest: nest.energy)
            configuration[word_node_id] = {
                "word": word_data["word"],
                "instance_id": word_data["instance_id"],
                "synset": best_nest.synset,
                "definition": best_nest.definition,
                "energy": best_nest.energy
            }
        return configuration

    def compute_global_score(self, configuration):
        selected_nests = []
        for word_node_id, word_data in self.word_to_nests.items():
            chosen_synset = configuration[word_node_id]["synset"]
            for nest in word_data["nests"]:
                if nest.synset == chosen_synset:
                    selected_nests.append(nest)
                    break

        score = 0
        for i in range(len(selected_nests)):
            for j in range(i + 1, len(selected_nests)):
                score = score + ext_lesk_nests(selected_nests[i].odour_vector, selected_nests[j].odour_vector)
        return score

    def update_best_configuration(self):
        configuration = self.get_current_configuration()
        score = self.compute_global_score(configuration)
        if score > self.best_score:
            self.best_score = score
            self.best_configuration = configuration

    def run(self):
        for cycle in range(C_AC):
            if cycle % 10 == 0:
                print(f"  Cycle {cycle}/{C_AC}, ants alive: {len(self.ants)}")
            self.remove_dead_ants()
            self.remove_collapsed_bridges()
            self.produce_ants()
            self.move_ants()
            for ant in self.ants:
                self.deposit_odour_on_node(ant)
                self.try_create_bridge(ant)
            self.evaporate_pheromones()
            self.update_best_configuration()
        return self.best_configuration


# EVALUATION

def synset_to_sense_keys(synset):
    return [lemma.key() for lemma in synset.lemmas()]

def evaluate_with_gold(result, gold):
    total_predicted = 0
    correct = 0
    total_gold = 0

    total_gold = len(gold)

    for _, data in result.items():
        instance_id = data["instance_id"]

        if instance_id is None:
            continue

        if instance_id not in gold:
            continue

        total_predicted += 1

        predicted_keys = synset_to_sense_keys(data["synset"])
        gold_keys = gold[instance_id]

        if any(key in gold_keys for key in predicted_keys):
            correct += 1

    precision = correct / total_predicted if total_predicted > 0 else 0
    recall = correct / total_gold if total_gold > 0 else 0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    accuracy = correct / total_predicted if total_predicted > 0 else 0

    return {
        "total_gold": total_gold,
        "total_predicted": total_predicted,
        "correct": correct,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# MAIN

if __name__ == "__main__":
    xml_path = "data/semeval2007/eng-coarse-all-words.xml"
    key_path = "data/semeval2007/dataset21.test.key"

    semeval_texts = load_semeval_xml(xml_path)
    gold = load_gold_key(key_path)

    # processed_sentences = preprocess_semeval_text(semeval_texts[0])

    # limited_sentences = []
    # word_count = 0
    # for sentence in processed_sentences:
    #     new_sentence = []
    #     for word_data in sentence:
    #         # if word_count >= 500:
    #         #     break
    #         new_sentence.append(word_data)
    #         word_count = word_count + 1
    #     if new_sentence:
    #         limited_sentences.append(new_sentence)
    #     # if word_count >= 500:
    #     #     break

    # processed_sentences = limited_sentences

    processed_sentences = []

    for semeval_text in semeval_texts:
        text_sentences = preprocess_semeval_text(semeval_text)
        processed_sentences.extend(text_sentences)

    print("Building graph...")
    graph_builder = GraphBuilder()
    graph_data = graph_builder.build_graph(processed_sentences)
    print(f"Graph built: {len(graph_data['nodes'])} nodes, {len(graph_data['edges'])} edges")

    print("Running ant colony...")
    colony = AntColony(graph_data)
    result = colony.run()

    print("\nFirst 20 selected senses:\n")
    for _, data in list(result.items())[:20]:
        print(f"ID: {data['instance_id']}")
        print(f"Word: {data['word']}")
        print(f"Sense: {data['synset'].name()}")
        print(f"Definition: {data['definition']}")
        print(f"Energy: {data['energy']}")
        print("-" * 50)

    print("Best global score:", colony.best_score)
    print("Number of nodes:", len(graph_data["nodes"]))
    print("Number of edges:", len(graph_data["edges"]))
    print("Number of ants created:", colony.ant_counter)

    gold_sample = list(gold.keys())[:3]
    result_ids = [d["instance_id"] for d in result.values() if d["instance_id"]][:3]
    print("\nGold key sample:  ", gold_sample)
    print("Result ID sample: ", result_ids)

    metrics = evaluate_with_gold(result, gold)

    print("\nEvaluation:")
    print("Total gold:", metrics["total_gold"])
    print("Total predicted:", metrics["total_predicted"])
    print("Correct:", metrics["correct"])
    print("Accuracy:", round(metrics["accuracy"], 4))
    print("Precision:", round(metrics["precision"], 4))
    print("Recall:", round(metrics["recall"], 4))
    print("F1:", round(metrics["f1"], 4))
