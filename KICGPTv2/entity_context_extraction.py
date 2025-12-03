"""
Entity Context Extraction for KICGPTv2
Extracts entity contexts (t_e, tau_e) using LLM for all entities in the KG
"""
import argparse
import json
import logging
import os
import time
import openai
from tqdm import tqdm
from collections import defaultdict
import multiprocessing as mp


class ChatGPT:
    def __init__(self, args, prompt_path, prompt_name, max_tokens):
        self.args = args
        self.history_messages = []
        self.max_tokens = max_tokens
        self.prompt = self.load_prompt_template(prompt_path, prompt_name)
        
    def get_response(self, input_text, turn_type):
        message = self.create_message(input_text, turn_type)
        if self.args.debug:
            print("Query API to get message:\n%s" % message['content'])
            response = input("Input the returned response:")
        else:
            self.history_messages.append(message)
            response_message = self.query_API_to_get_message(self.history_messages)
            self.history_messages.append(response_message)
            response = response_message['content'].strip()
        return response
    
    def create_message(self, input_text, turn_type):
        if turn_type == "system_prompt":
            template = self.prompt['system_prompt']
            entity_name = input_text
            input_text = template.format(entity=entity_name)
        elif turn_type == "corresponding_paragraph":
            template = self.prompt['corresponding_paragraph']
            paragraph = input_text
            input_text = template.format(paragraph=paragraph)
        elif turn_type == "summarization_prompt":
            template = self.prompt['summarization_prompt']
            entity_name = input_text
            input_text = template.format(entity=entity_name)
        else:
            raise NotImplementedError
        return {'role': 'user', 'content': input_text}
    
    def query_API_to_get_message(self, messages):
        while True:
            try:
                res = openai.ChatCompletion.create(
                    model="Qwen/Qwen2.5-72B-Instruct",  
                    messages=messages,
                    temperature=0,
                    max_tokens=self.max_tokens,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                return res['choices'][0]['message']
            except openai.error.RateLimitError:
                print('openai.error.RateLimitError\nRetrying...')
                time.sleep(30)
            except openai.error.ServiceUnavailableError:
                print('openai.error.ServiceUnavailableError\nRetrying...')
                time.sleep(20)
            except openai.error.Timeout:
                print('openai.error.Timeout\nRetrying...')
                time.sleep(20)
            except openai.error.APIError:
                print('openai.error.APIError\nRetrying...')
                time.sleep(20)
            except openai.error.APIConnectionError:
                print('openai.error.APIConnectionError\nRetrying...')
                time.sleep(20)
    
    def reset_history(self):
        self.history_messages = []
    
    def load_prompt_template(self, prompt_path, prompt_name):
        if prompt_path.endswith(".json"):
            with open(prompt_path, "rb") as f:
                prompt = json.load(f)
            return prompt[prompt_name]


class EntityContextExtractor:
    def __init__(self, args):
        self.args = args
        self.LLM = ChatGPT(args=args, prompt_path=args.prompt_path, 
                          prompt_name=args.prompt_name, max_tokens=args.max_tokens)
        self.ent2text = defaultdict(str)
        self.id2ent = defaultdict(str)
        self.entity_neighbors = defaultdict(list)
        
        self.load_ent_to_text()
        self.load_ent_map_id()
        self.load_entity_neighbors()
    
    def extract_context(self, entity):
        """Extract context (t_e, tau_e) for an entity using Appendix format"""
        self.LLM.reset_history()
        entity_text = self.ent2text.get(entity, entity)
        
        # Get related triples for this entity to create a paragraph
        neighbors = self.entity_neighbors.get(entity, [])
        paragraph = self.create_paragraph(entity_text, neighbors[:10])
        
        # Step 1: System prompt
        self.LLM.get_response(entity_text, "system_prompt")
        
        # Step 2: Provide paragraph
        self.LLM.get_response(paragraph, "corresponding_paragraph")
        
        # Step 3: Request summarization
        context = self.LLM.get_response(entity_text, "summarization_prompt")
        
        # Parse output: [short description: ...] [aliases:{...}]
        description, aliases = self.parse_context(context)
        
        return {
            "entity": entity,
            "entity_text": entity_text,
            "description": description,
            "aliases": aliases,
            "raw_context": context
        }
    
    def create_paragraph(self, entity_text, triples):
        """Create a paragraph describing the entity based on triples"""
        if not triples:
            return f"{entity_text} is an entity in the knowledge graph."
        
        sentences = []
        for h, r, t in triples[:5]:  # Use top 5 for paragraph
            h_text = self.ent2text.get(h, h)
            t_text = self.ent2text.get(t, t)
            sentences.append(f"{h_text} {r} {t_text}")
        
        paragraph = entity_text + ". " + ". ".join(sentences) + "."
        return paragraph
    
    def parse_context(self, context):
        """Parse LLM output to extract description and aliases"""
        import re
        
        # Extract description
        desc_match = re.search(r'\[short description:\s*([^\]]+)\]', context)
        description = desc_match.group(1).strip() if desc_match else context.split('[aliases')[0].strip()
        
        # Extract aliases
        alias_match = re.search(r'\[aliases:\s*\{([^}]*)\}\]', context)
        if alias_match:
            aliases_str = alias_match.group(1)
            aliases = [a.strip() for a in aliases_str.split(',') if a.strip()]
        else:
            aliases = []
        
        return description, aliases
    
    def load_ent_to_text(self):
        with open('datasets/' + self.args.dataset + '/entity2text.txt', 'r') as file:
            for line in file:
                ent, text = line.strip().split("\t")
                self.ent2text[ent] = text
    
    def load_ent_map_id(self):
        with open('datasets/' + self.args.dataset + '/get_neighbor/entity2id.txt', 'r') as file:
            for line in file:
                _name, _id = line.strip().split("\t")
                self.id2ent[_id] = _name
    
    def load_entity_neighbors(self):
        """Load entity neighbors from demonstration files"""
        try:
            with open("datasets/" + self.args.dataset + "/demonstration/tail_supplement.txt", "r") as f:
                tail_supplement = json.load(f)
            with open("datasets/" + self.args.dataset + "/demonstration/head_supplement.txt", "r") as f:
                head_supplement = json.load(f)
            
            # Combine both to get all entity neighbors
            for key, triples in tail_supplement.items():
                entity = key.split('\t')[0]
                self.entity_neighbors[entity].extend(triples)
            
            for key, triples in head_supplement.items():
                entity = key.split('\t')[0]
                self.entity_neighbors[entity].extend(triples)
        except:
            print("Warning: Could not load entity neighbors")


def main(args, all_entities, idx, api_key):
    import openai
    openai.api_key = api_key
    # Configure API base for Qwen-2.5-72B as per reviewer requirement
    openai.api_base = 'https://api.pumpkinaigc.online/v1'
    
    if idx == -1:
        output_path = args.output_path
    else:
        idx = "0" + str(idx) if idx < 10 else str(idx)
        output_path = args.output_path + "_" + idx
    
    print("Start PID %d and save to %s" % (os.getpid(), output_path))
    extractor = EntityContextExtractor(args)
    
    count = 0
    with open(output_path, "w") as f:
        for entity in tqdm(all_entities, total=len(all_entities)):
            count += 1
            try:
                context_info = extractor.extract_context(entity)
                f.write(json.dumps(context_info) + "\n")
            except Exception as e:
                logging.exception(e)
                continue
    
    print("---------------PID %d end with %d samples--------------" % (os.getpid(), count))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="fb15k-237")
    parser.add_argument('--output_path', default="./datasets/fb15k-237/entity_contexts.txt")
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--max_tokens', default=300, type=int)
    parser.add_argument('--prompt_path', default="./prompts/entity_context.json")
    parser.add_argument('--prompt_name', default="chat")
    parser.add_argument('--api_key', default="", type=str, help='API key for LLM access')
    parser.add_argument('--model', default="Qwen/Qwen2.5-72B-Instruct", type=str, 
                       help='LLM model name (default: Qwen-2.5-72B)')
    parser.add_argument('--num_process', default=1, type=int)
    
    args = parser.parse_args()
    args.output_path = './datasets/' + args.dataset + '/entity_contexts.txt'
    return args


if __name__ == '__main__':
    args = parse_args()
    
    if not args.api_key.startswith("sk-"):
        with open(args.api_key, "r") as f:
            all_keys = f.readlines()
            all_keys = [line.strip('\n') for line in all_keys]
            assert len(all_keys) == args.num_process
    
    # Load all entities
    all_entities = []
    with open('datasets/' + args.dataset + '/get_neighbor/entity2id.txt', 'r') as file:
        for line in file:
            _name, _id = line.strip().split("\t")
            all_entities.append(_name)
    
    print("Totally %d entities." % len(all_entities))
    
    if args.num_process == 1:
        main(args, all_entities, idx=-1, api_key=args.api_key)
    else:
        num_each_split = int(len(all_entities) / args.num_process)
        p = mp.Pool(args.num_process)
        for idx in range(args.num_process):
            start = idx * num_each_split
            if idx == args.num_process - 1:
                end = max((idx + 1) * num_each_split, len(all_entities))
            else:
                end = (idx + 1) * num_each_split
            split_data = all_entities[start:end]
            try:
                p.apply_async(main, args=(args, split_data, idx, all_keys[idx]))
            except Exception as e:
                logging.exception(e)
        p.close()
        p.join()
        print("All of the child processes over!")
        
        # Merge split files into single entity_contexts.txt
        print("Merging split files...")
        with open(args.output_path, 'w') as outfile:
            for idx in range(args.num_process):
                idx_str = "0" + str(idx) if idx < 10 else str(idx)
                split_file = args.output_path + "_" + idx_str
                if os.path.exists(split_file):
                    with open(split_file, 'r') as infile:
                        outfile.write(infile.read())
                    os.remove(split_file)  # Clean up split file
                    print(f"Merged {split_file}")
        print(f"All entity contexts saved to {args.output_path}")
