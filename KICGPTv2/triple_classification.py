import argparse
import json
import logging
import os
import re
import time
import openai
from tqdm import tqdm 
import multiprocessing as mp
from collections import defaultdict


class ChatGPT:
    def __init__(self, args, prompt_path, prompt_name, max_tokens):
        self.args = args
        self.history_messages = []
        self.history_contents = []
        self.max_tokens = max_tokens
        self.prompt = self.load_prompt_template(prompt_path, prompt_name)
        self.token_num = 0
    
    def get_response(self, input_text, turn_type):
        if self.args.debug:
            message = self.create_message(input_text, turn_type)
            self.history_messages.append(message)
            self.history_contents.append(message['content'])
            print("query API to get message:\n%s" % message['content'])
            response = input("input the returned response:")
        else:
            message = self.create_message(input_text, turn_type)
            self.history_messages.append(message)
            self.history_contents.append(message['content'])
            message = self.query_API_to_get_message(self.history_messages)
            self.history_messages.append(message)
            self.history_contents.append(message['content'])
            response = message['content'].strip()
        return response

    def create_message(self, input_text, turn_type):
        if turn_type == "init_query":  
            instruction = self.prompt['init_query']
            input_text = instruction

        elif turn_type == "statement_demonstration":
            template = self.prompt['statement_demonstration']
            statement, entity_contexts = input_text
            input_text = template.format(statement=statement, entity_contexts=entity_contexts)
        elif turn_type == "analogy_fact":
            template = self.prompt['analogy_fact']
            analogy_facts = input_text
            input_text = template.format(analogy_facts=analogy_facts)
        elif turn_type == "supplement_fact":
            template = self.prompt['supplement_fact']
            supplement_facts = input_text
            input_text = template.format(supplement_facts=supplement_facts)
        elif turn_type == "final_judgment":
            template = self.prompt['final_judgment']
            entity_contexts = input_text
            input_text = template.format(entity_contexts=entity_contexts)
        else:
            raise NotImplementedError
        message = {'role': 'user', 'content': input_text}
        return message

    def query_API_to_get_message(self, messages):
        while True:
            try:
                res = openai.ChatCompletion.create(
                    model="Qwen/Qwen2.5-72B-Instruct",  # Qwen-2.5-72B
                    messages=messages,
                    temperature=0,
                    max_tokens=self.max_tokens,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                if args.debug_online:
                    print(res)
                self.token_num = res['usage']['total_tokens']
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
        self.history_contents = []
        self.token_num = 0

    def load_prompt_template(self, prompt_path, prompt_name):
        if prompt_path.endswith(".json"):
            with open(prompt_path, "rb") as f:
                prompt = json.load(f)
            return prompt[prompt_name]


import tiktoken


class Solver:
    def __init__(self, args):
        self.args = args
        self.LLM = ChatGPT(args=args, prompt_path=args.prompt_path, prompt_name=args.prompt_name,
                           max_tokens=args.max_tokens)
        self.max_llm_input_token = args.max_llm_input_tokens
        
        self.log = []
        
        self.id2ent = defaultdict(str)
        self.ent2id = defaultdict(str)
        self.rel2id = defaultdict(str)
        self.id2rel = defaultdict(str)
        self.ent2text = defaultdict(str)
        self.rel2text = defaultdict(str)
        self.entity_contexts = defaultdict(dict)
        self.refined_relations = defaultdict(str)
        
        # For triple classification, we need the training graph
        self.train_triples = set()
        
        self.load_rel_txt_to_id()
        self.load_ent_map_id()
        self.load_ent_to_text()
        self.load_rel_to_text()
        self.load_train_triples()
        if self.args.use_entity_context:
            self.load_entity_contexts()
        if self.args.use_relation_alignment:
            self.load_refined_relations()
        
    def count_token(self, string):
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0613")
        return len(encoding.encode(string))
    
    def get_entity_context_text(self, entity):
        """Get entity context in Appendix format: description + aliases"""
        if entity in self.entity_contexts:
            ctx = self.entity_contexts[entity]
            entity_text = ctx.get('entity_text', entity)
            description = ctx.get('description', ctx.get('context', ''))  # Support both new and old format
            aliases = ctx.get('aliases', [])
            
            # Format: "Entity is description. It has aliases: {alias1, alias2}"
            text = f"{entity_text} is {description}" if description else entity_text
            if aliases and isinstance(aliases, list):
                alias_str = ", ".join(aliases)
                text += f". It has aliases: {{{alias_str}}}"
            return text
        return self.ent2text.get(entity, entity)
                
    def forward(self, head_entity, relation, tail_entity):
        """
        Classify whether triple (h, r, t) is valid
        Returns: "Yes" or "No"
        """
        self.LLM.reset_history()
        self.reset_history()
        
        head_str = self.ent2text.get(head_entity, head_entity)
        tail_str = self.ent2text.get(tail_entity, tail_entity)
        rel_str = self.rel2text.get(relation, relation)
        
        # Get entity contexts
        entity_contexts_text = ""
        if self.args.use_entity_context:
            head_ctx = self.get_entity_context_text(head_entity)
            tail_ctx = self.get_entity_context_text(tail_entity)
            entity_contexts_text = f"{head_str} is {head_ctx}. {tail_str} is {tail_ctx}."
        
        # Generate statement
        statement_text = self.generate_statement_text(head_str, rel_str, tail_str, entity_contexts_text)
        
        query_token_num = self.count_token(
            self.LLM.create_message((entity_contexts_text,), "final_judgment")['content'])

        init_response = self.LLM.get_response((''), "init_query")
        assert self.check_work_flow(init_response), "LLM Not Understand Task"
        
        # Present the statement
        statement_response = self.LLM.get_response((statement_text, entity_contexts_text), 
                                                   "statement_demonstration")
        
        effective_demon_step = 0
        current_demon_step = 0

        # Provide demonstrations (known facts)
        while effective_demon_step < args.eff_demon_step and current_demon_step < args.max_demon_step:
            analogy_facts, supplement_facts = self.get_demonstrations(
                head_entity, relation, tail_entity, args.demon_per_step, current_demon_step)
            
            analogy_facts_text = self.serialize_facts(analogy_facts)
            supplement_facts_text = self.serialize_facts(supplement_facts)
            
            if analogy_facts_text == "None." and supplement_facts_text == "None.": 
                break

            if analogy_facts_text != "None.":
                current_demon_response = self.LLM.get_response((analogy_facts_text), "analogy_fact")
                if self.LLM.token_num >= args.max_llm_input_tokens - query_token_num: 
                    self.LLM.history_messages.pop()
                    self.LLM.history_messages.pop()
                    self.LLM.history_contents.pop()
                    self.LLM.history_contents.pop()
                    break
                    
            if supplement_facts_text != "None.":
                current_demon_response = self.LLM.get_response((supplement_facts_text), "supplement_fact")
                if self.LLM.token_num >= args.max_llm_input_tokens - query_token_num: 
                    self.LLM.history_messages.pop()
                    self.LLM.history_messages.pop()
                    self.LLM.history_contents.pop()
                    self.LLM.history_contents.pop()
                    break
                    
            current_demon_step += 1
            
            if self.check_work_flow(current_demon_response): 
                effective_demon_step += 1
                     
            self.log.append(f'demonstration: {effective_demon_step:02d}/{current_demon_step:02d} step')
            print(f'demonstration: {effective_demon_step:02d}/{current_demon_step:02d} step')
            
        # Final judgment
        final_response = self.LLM.get_response((entity_contexts_text,), "final_judgment")
        
        self.log.append(final_response)
        
        # Parse result - should be "Yes" or "No"
        judgment = self.parse_result(final_response)
        
        self.log.append(judgment)
        
        return judgment, self.LLM.history_contents, self.log
    
    def check_work_flow(self, response):
        if "no" in response.lower() and "yes" not in response.lower():
            return False
        return True
    
    def generate_statement_text(self, head, relation, tail, entity_contexts):
        """Generate statement for triple classification"""
        statement = f"{head} {relation} {tail}. The judgment is: ?"
        if entity_contexts:
            statement = statement + f" {entity_contexts}"
        return statement
    
    def serialize_facts(self, facts):
        """Serialize known facts into text"""
        fact_text = ""
        for h, r, t in facts:
            h_text = self.ent2text.get(h, h)
            t_text = self.ent2text.get(t, t)
            r_text = self.rel2text.get(r, r)
            fact_text += f"{h_text} {r_text} {t_text}. "
        fact_text = fact_text.strip()
        if fact_text == "": 
            fact_text = "None."
        return fact_text
    
    def get_demonstrations(self, head, relation, tail, num, step):
        """
        Get demonstration facts
        - Analogy: facts with same relation
        - Supplement: facts involving head or tail entity
        """
        analogy_facts = []
        supplement_facts = []
        
        # Get facts with same relation (analogy)
        for triple in list(self.train_triples)[step*num//2:(step+1)*num//2]:
            if triple[1] == relation and (triple[0] != head or triple[2] != tail):
                analogy_facts.append(triple)
        
        # Get facts involving head or tail (supplement)
        for triple in list(self.train_triples)[step*num:(step+1)*num]:
            if (triple[0] == head or triple[2] == head or 
                triple[0] == tail or triple[2] == tail):
                if triple != (head, relation, tail):
                    supplement_facts.append(triple)
        
        return analogy_facts[:num//2], supplement_facts[:num//2]
         
    def parse_result(self, response):
        """Parse LLM response to get Yes or No"""
        response_lower = response.lower()
        
        # Look for final judgment
        if "the final judgment:" in response_lower:
            judgment_part = response_lower.split("the final judgment:")[1]
        else:
            judgment_part = response_lower
        
        # Check which comes first
        yes_pos = judgment_part.find("yes")
        no_pos = judgment_part.find("no")
        
        if yes_pos != -1 and (no_pos == -1 or yes_pos < no_pos):
            return "Yes"
        elif no_pos != -1:
            return "No"
        else:
            # Default to No if unclear
            return "No"
        
    def reset_history(self):
        self.log = []
    
    def load_train_triples(self):
        """Load training triples for finding demonstrations"""
        with open('datasets/' + self.args.dataset + '/get_neighbor/train2id.txt', 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 3:
                    head_id, rel_id, tail_id = parts
                    head = self.id2ent.get(head_id, head_id)
                    rel = self.id2rel.get(rel_id, rel_id)
                    tail = self.id2ent.get(tail_id, tail_id)
                    self.train_triples.add((head, rel, tail))
    
    def load_entity_contexts(self):
        """Load entity contexts (description + aliases)"""
        try:
            with open("datasets/" + self.args.dataset + "/entity_contexts.txt", 'r') as f:
                for line in f:
                    ctx_data = json.loads(line.strip())
                    entity = ctx_data.get('entity')
                    if entity:
                        self.entity_contexts[entity] = ctx_data
                print(f"Loaded {len(self.entity_contexts)} entity contexts")
        except FileNotFoundError:
            print("Warning: entity_contexts.txt not found. Run entity_context_extraction.py first.")
    
    def load_refined_relations(self):
        """Load precomputed refined relations"""
        try:
            with open("datasets/" + self.args.dataset + "/alignment/alignment_clean.txt", 'r') as f:
                self.refined_relations = json.load(f)
        except FileNotFoundError:
            print("Warning: Refined relations file not found.")
            
    def load_rel_txt_to_id(self):
        with open('datasets/' + self.args.dataset + '/get_neighbor/relation2id.txt', 'r') as file:
            for line in file:
                _name, _id = line.strip().split("\t")
                self.rel2id[_name] = _id
                self.id2rel[_id] = _name
                
    def load_ent_map_id(self):
        with open('datasets/' + self.args.dataset + '/get_neighbor/entity2id.txt', 'r') as file:
            for line in file:
                _name, _id = line.strip().split("\t")
                self.ent2id[_name] = _id
                self.id2ent[_id] = _name
    
    def load_ent_to_text(self):
        with open('datasets/' + self.args.dataset + '/entity2text.txt', 'r') as file:
            for line in file:
                ent, text = line.strip().split("\t")
                self.ent2text[ent] = text
    
    def load_rel_to_text(self):
        try:
            with open('datasets/' + self.args.dataset + '/relation2text.txt', 'r') as file:
                for line in file:
                    rel, text = line.strip().split("\t")
                    self.rel2text[rel] = text
        except FileNotFoundError:
            print("Warning: relation2text.txt not found. Using relation names directly.")
            for rel in self.rel2id.keys():
                self.rel2text[rel] = rel


def main(args, all_data, idx, api_key):
    import openai
    openai.api_key = api_key
    # Configure API base for Qwen-2.5-72B
    openai.api_base = 'https://api.pumpkinaigc.online/v1'
    
    if idx == -1:
        output_path = args.output_path
        chat_log_path = args.chat_log_path
    else:
        idx = "0" + str(idx) if idx < 10 else str(idx)
        output_path = args.output_path + "_" + idx
        chat_log_path = args.chat_log_path + "_" + idx

    print("Start PID %d and save to %s" % (os.getpid(), output_path))
    solver = Solver(args)

    count = 0
    valid_count = 0
    correct_count = 0
    
    with open(output_path, "w") as f:
        with open(chat_log_path, "w") as fclog:
            for sample in tqdm(all_data, total=len(all_data)):
                count += 1
                try:
                    head_entity = sample['HeadEntity']
                    relation = sample['Relation']
                    tail_entity = sample['TailEntity']
                    true_label = sample['Label']  # "Yes" or "No"
                    
                    prediction, chat_history, record = solver.forward(head_entity, relation, tail_entity)
                    valid_count += 1
                    
                    # Check if prediction is correct
                    if prediction == true_label:
                        correct_count += 1
                        
                except openai.error.InvalidRequestError as e:
                    print(e)
                    continue
                except Exception as e:
                    logging.exception(e)
                    continue

                chat = str(sample["ID"]) + "\n" + "\n******\n".join(chat_history) + "\nTrue Label: " + str(
                    true_label) + "\n------------------------------------------\n"
                fclog.write(chat)

                sample["Prediction"] = prediction
                f.write(json.dumps(sample) + "\n")

    accuracy = correct_count / valid_count if valid_count > 0 else 0
    print("---------------PID %d end with %d/%d samples, Accuracy: %.2f%%--------------" % 
          (os.getpid(), valid_count, count, accuracy * 100))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="fb13")
    parser.add_argument('--output_path', default="./outputs/fb13/output_classification.txt")
    parser.add_argument('--chat_log_path', default="./outputs/fb13/chat_classification.txt")
    
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--debug_online', action="store_true")
    
    # KICGPTv2 specific arguments
    parser.add_argument('--use_entity_context', action="store_true", 
                       help='Use entity context extraction (KICGPTv2)')
    parser.add_argument('--use_relation_alignment', action="store_true",
                       help='Use relation self-alignment (KICGPTv2)')
    # Note: use_reconstruction not needed for triple classification (binary answers always match)
    
    parser.add_argument('--max_tokens', default=300, type=int, help='max-token')
    parser.add_argument('--prompt_path', default="./prompts/triple_classification.json")
    parser.add_argument('--prompt_name', default="chat")
    parser.add_argument('--overwrite', action="store_true")
    parser.add_argument('--device', default=0, help='the gpu device')
    
    parser.add_argument('--api_key', default="", type=str)
    parser.add_argument('--demon_per_step', default=8, type=int)
    parser.add_argument('--eff_demon_step', default=32, type=int, help='Default 32 as per Appendix B')
    parser.add_argument('--max_demon_step', default=32, type=int)
    parser.add_argument('--max_llm_input_tokens', default=3750, type=int)
    parser.add_argument('--num_process', default=1, type=int, help='the number of multi-process')

    args = parser.parse_args()
    args.output_path = './outputs/' + args.dataset + '/output_classification.txt'
    args.chat_log_path = './outputs/' + args.dataset + '/chat_classification.txt'
    print("Start querying the LLM for triple classification.")
    return args


if __name__ == '__main__':
    args = parse_args()
    if not args.api_key.startswith("sk-"):
        with open(args.api_key, "r") as f:
            all_keys = f.readlines()
            all_keys = [line.strip('\n') for line in all_keys]
            assert len(all_keys) == args.num_process, (len(all_keys), args.num_process)
    
    test_triplet = []

    # Load test data for triple classification
    with open("datasets/" + args.dataset + "/test_classification.txt", 'r') as load_f:
        test_triplet = json.load(load_f)
    print("Totally %d test examples." % len(test_triplet))

    if args.debug_online:
        test_triplet = test_triplet[0:2*args.num_process]
    if args.num_process == 1:
        main(args, test_triplet, idx=-1, api_key=args.api_key)
    else:
        num_each_split = int(len(test_triplet) / args.num_process)
        p = mp.Pool(args.num_process)
        for idx in range(args.num_process):
            start = idx * num_each_split
            if idx == args.num_process - 1:
                end = max((idx + 1) * num_each_split, len(test_triplet))
            else:
                end = (idx + 1) * num_each_split
            split_data = test_triplet[start:end]
            try:
                p.apply_async(main, args=(args, split_data, idx, all_keys[idx]))
            except Exception as e:
                logging.exception(e)

        p.close()
        p.join()
        print("All of the child processes over!")

# Usage:
# python triple_classification.py --dataset fb13 --use_entity_context --api_key <key>
