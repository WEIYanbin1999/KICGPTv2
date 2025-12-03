import argparse
import json
import logging
import os
import re
import time
import openai
from tqdm import tqdm 
import multiprocessing as mp
from prompt_selection import Demon_sampler
from difflib import SequenceMatcher


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

        elif turn_type == "first_give_demonstration":
            template = self.prompt['first_give_demonstration']
            question, entity_contexts = input_text
            input_text = template.format(question=question, entity_contexts=entity_contexts)
        elif turn_type == "analogy_demonstration":
            template = self.prompt['analogy_demonstration']
            analogy_demons = input_text
            input_text = template.format(selected_analogy_demonstrations=analogy_demons)
        elif turn_type == "supplement_demonstration":
            template = self.prompt['supplement_demonstration']
            supplement_demons = input_text
            input_text = template.format(selected_supplement_demonstrations=supplement_demons)
        elif turn_type == "final_query_template":
            template = self.prompt['final_query_template']
            can_rels, question, entity_contexts = input_text
            input_text = template.format(order_of_candidate=can_rels, question=question, entity_contexts=entity_contexts)
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


from collections import defaultdict
import tiktoken


class Solver:
    def __init__(self, args):
        self.args = args
        self.LLM = ChatGPT(args=args, prompt_path=args.prompt_path, prompt_name=args.prompt_name,
                           max_tokens=args.max_tokens)
        self.max_llm_input_token = args.max_llm_input_tokens
        self.prompt_selector = Demon_sampler(args)
        
        self.log = []
        self.candidate_answers = []
        self.selected_demonstrations = []
        
        self.id2ent = defaultdict(str)
        self.ent2id = defaultdict(str)
        self.rel2id = defaultdict(str)
        self.id2rel = defaultdict(str)
        self.ent2text = defaultdict(str)
        self.rel2text = defaultdict(str)
        self.all_candidate_answers = defaultdict(list)
        self.entity_contexts = defaultdict(dict)
        self.refined_relations = defaultdict(str)
        
        self.load_rel_txt_to_id()
        self.load_ent_map_id()
        self.load_all_candidate_answers()
        self.load_ent_to_text()
        self.load_rel_to_text()
        if self.args.use_entity_context:
            self.load_entity_contexts()
        if self.args.use_relation_alignment:
            self.load_refined_relations()
        
    def count_token(self, string):
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0613")
        return len(encoding.encode(string))
    
    def compute_similarity(self, text1, text2):
        """Compute similarity between two texts using SequenceMatcher"""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def retrieval_augmented_reconstruction(self, llm_output, candidate_answers, beta=0.33):
        """
        Implements retrieval-augmented reconstruction for relation prediction
        """
        L_retrieval = []
        seen = set()
        
        # Parse LLM output to get ordered list
        llm_ordered = self.parse_llm_output_to_list(llm_output)
        
        # For each LLM answer, find most similar candidate
        for llm_answer in llm_ordered:
            best_match = None
            best_score = 0
            
            for candidate in candidate_answers:
                if candidate in seen:
                    continue
                    
                # Compute similarity score
                sim_score = self.compute_similarity(llm_answer, candidate)
                
                if sim_score > best_score:
                    best_score = sim_score
                    best_match = candidate
            
            # Only add if similarity exceeds threshold beta
            if best_match and best_score > beta:
                L_retrieval.append(best_match)
                seen.add(best_match)
        
        # Add remaining candidates from preliminary ordering
        for candidate in candidate_answers:
            if candidate not in seen:
                L_retrieval.append(candidate)
        
        return L_retrieval
    
    def parse_llm_output_to_list(self, llm_output):
        """Parse LLM output string to ordered list of answers"""
        llm_output = llm_output.lower()
        if "the final order:" in llm_output:
            llm_output = llm_output.split("the final order:")[1].strip()
        
        # Remove brackets and split by delimiter
        llm_output = llm_output.strip('.').strip('[').strip(']')
        answers = [ans.strip() for ans in llm_output.split('|')]
        
        return answers
    
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
                
    def forward(self, head_entity, tail_entity):
        """
        Predict relation between head and tail entities
        Args:
            head_entity: head entity ID
            tail_entity: tail entity ID
        """
        self.LLM.reset_history()
        self.reset_history()
        
        head_str = self.ent2text[head_entity]
        tail_str = self.ent2text[tail_entity]
        
        # Get candidate relations (all relations in dataset)
        candidate_rel_ids = self.all_candidate_answers['\t'.join([str(self.ent2id[head_entity]), str(self.ent2id[tail_entity])])]
        for rid in candidate_rel_ids[:args.candidate_num]:
            rel_name = self.id2rel.get(str(rid), str(rid))
            rel_text = self.rel2text.get(rel_name, rel_name)
            self.candidate_answers.append(rel_text)
        
        origin_candidates_text = self.serialize_candidate_answers()
        
        # Get entity contexts for head and tail
        entity_contexts_text = ""
        if self.args.use_entity_context:
            head_ctx = self.get_entity_context_text(head_entity)
            tail_ctx = self.get_entity_context_text(tail_entity)
            entity_contexts_text = f"Head entity: {head_ctx}. Tail entity: {tail_ctx}."
        
        # Generate question text
        question_text = self.generate_question_text(head_str, tail_str)
        
        query_token_num = self.count_token(
            self.LLM.create_message((origin_candidates_text, question_text, entity_contexts_text), 
                                   "final_query_template")['content'])

        init_response = self.LLM.get_response((''), "init_query")
        assert self.check_work_flow(init_response), "LLM Not Understand Task"
        
        effective_demon_step = 0
        current_demon_step = -1

        while effective_demon_step < args.eff_demon_step and current_demon_step < args.max_demon_step:
            if current_demon_step == -1:
                current_demon_response = self.LLM.get_response((question_text, entity_contexts_text), 
                                                               "first_give_demonstration")
                current_demon_step += 1
                # For relation prediction, use same relation examples (analogy)
                true_demons = self.get_analogy_demonstrations(head_entity, tail_entity, num=args.demon_per_step//2)
                true_demon_text = self.serialize_demonstrations(true_demons)
                if true_demon_text != "None.":
                    current_demon_response = self.LLM.get_response((true_demon_text), "analogy_demonstration")
                if self.LLM.token_num >= args.max_llm_input_tokens - query_token_num: 
                    self.LLM.history_messages.pop()
                    self.LLM.history_messages.pop()
                    self.LLM.history_contents.pop()
                    self.LLM.history_contents.pop()
                    break
                continue
            
            analogy_demons, supplement_demons = self.get_demonstrations(head_entity, tail_entity, 
                                                                        args.demon_per_step, current_demon_step)
            analogy_demon_text = self.serialize_demonstrations(analogy_demons)
            supplement_demon_text = self.serialize_demonstrations(supplement_demons)
            if analogy_demon_text == "None." and supplement_demon_text == "None.": 
                break

            if analogy_demon_text != "None.":
                current_demon_response = self.LLM.get_response((analogy_demon_text), "analogy_demonstration")
                if self.LLM.token_num >= args.max_llm_input_tokens - query_token_num: 
                    self.LLM.history_messages.pop()
                    self.LLM.history_messages.pop()
                    self.LLM.history_contents.pop()
                    self.LLM.history_contents.pop()
                    break
            if supplement_demon_text != "None.":
                current_demon_response = self.LLM.get_response((supplement_demon_text), "supplement_demonstration")
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
            
        # Knowledge-prompting Re-ranking
        final_response = self.LLM.get_response((origin_candidates_text, question_text, entity_contexts_text),
                                               "final_query_template")
        
        self.log.append(final_response)
        
        # Retrieval-augmented Reconstruction
        if self.args.use_reconstruction:
            final_order_list = self.retrieval_augmented_reconstruction(
                final_response, 
                self.candidate_answers, 
                beta=self.args.similarity_beta
            )
            final_order = ' | '.join(final_order_list)
        else:
            final_order = self.parse_result(final_response, "final_answer")
        
        self.log.append(final_order)
        
        return final_order, self.LLM.history_contents, self.log
        
    def serialize_candidate_answers(self):
        candidiate_str = '[' + ', '.join(self.candidate_answers) + ']'
        return candidiate_str
    
    def check_work_flow(self, response):
        if "no" in response.lower():
            return False
        return True
    
    def generate_question_text(self, head, tail):
        """Generate question for relation prediction"""
        return f"Predict the [MASK] by completing the sentence \"{head} [MASK] {tail}?\" [MASK] should be: ?"
    
    def serialize_demonstrations(self, demon_triples):
        demon_text = ""
        for h, r, t in demon_triples:
            h_text = self.ent2text.get(h, h)
            t_text = self.ent2text.get(t, t)
            r_text = self.rel2text.get(r, r)
            demon_text += f"Predict the [MASK] by completing the sentence \"{h_text} [MASK] {t_text}.\" [MASK] should be: {r_text}. "
        demon_text = demon_text.strip()
        if demon_text == "": 
            demon_text = "None."
        return demon_text
    
    def get_analogy_demonstrations(self, head, tail, num):
        """Get demonstrations for relation prediction"""
        # For relation prediction, get triples with different entities but potentially informative relations
        demonstrations = []
        # This is simplified - in practice, use demonstration pool
        return demonstrations
    
    def get_demonstrations(self, head, tail, num, step):
        """Get demonstrations - simplified version"""
        # In practice, implement proper demonstration selection
        return [], []
         
    def parse_result(self, response, parse_type):
        response = response.lower()
        if parse_type == "final_answer":
            if "the final order:" in response:
                final_order_raw = re.split("the final order:", response)[1].strip().strip('.').strip('\[').strip('\]')
                final_order_raw_list = final_order_raw.split(' | ')
                final_order_list = []
                for candidate in final_order_raw_list:
                    if candidate not in final_order_list:
                        final_order_list.append(candidate)
                final_order = ' | '.join(final_order_list)
        return final_order
        
    def reset_history(self):
        self.log = []
        self.candidate_answers = []
        self.selected_demonstrations = []
        
    def load_all_candidate_answers(self):
        try:
            with open("datasets/" + self.args.dataset + "/retriever_candidate_relation.txt", 'r') as load_f:
                self.all_candidate_answers = json.load(load_f)
        except FileNotFoundError:
            print("Warning: retriever_candidate_relation.txt not found. Using all relations.")
            # Fallback: use all relations as candidates
            self.all_candidate_answers = defaultdict(lambda: list(self.rel2id.values()))
    
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
            relation_lines = file.readlines()
            for line in relation_lines:
                _name, _id = line.strip().split("\t")
                self.rel2id[_name] = _id
                self.id2rel[_id] = _name
                
    def load_ent_map_id(self):
        with open('datasets/' + self.args.dataset + '/get_neighbor/entity2id.txt', 'r') as file:
            entity_lines = file.readlines()
            for line in entity_lines:
                _name, _id = line.strip().split("\t")
                self.ent2id[_name] = _id
                self.id2ent[_id] = _name
    
    def load_ent_to_text(self):
        with open('datasets/' + self.args.dataset + '/entity2text.txt', 'r') as file:
            entity_lines = file.readlines()
            for line in entity_lines:
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
    with open(output_path, "w") as f:
        with open(chat_log_path, "w") as fclog:
            for sample in tqdm(all_data, total=len(all_data)):
                count += 1
                try:
                    head_entity = sample['HeadEntity']
                    tail_entity = sample['TailEntity']
                    
                    prediction, chat_history, record = solver.forward(head_entity, tail_entity)
                    valid_count += 1
                except openai.error.InvalidRequestError as e:
                    print(e)
                    continue
                except Exception as e:
                    logging.exception(e)
                    continue

                chat = str(sample["ID"]) + "\n" + "\n******\n".join(chat_history) + "\nCorrect Relation: " + str(
                    sample['Relation']) + "\n------------------------------------------\n"
                fclog.write(chat)

                sample["Prediction"] = prediction
                f.write(json.dumps(sample) + "\n")

    print("---------------PID %d end with %d/%d samples--------------" % (os.getpid(), valid_count, count))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="fb15k-237")
    parser.add_argument('--candidate_num', default=10, type=int, help='Number of candidate relations (default 10 as per manuscript)')
    parser.add_argument('--output_path', default="./outputs/fb15k-237/output_relation.txt")
    parser.add_argument('--chat_log_path', default="./outputs/fb15k-237/chat_relation.txt")
    parser.add_argument('--model_path', default=None)
    
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--debug_online', action="store_true")
    
    # KICGPTv2 specific arguments
    parser.add_argument('--use_entity_context', action="store_true", 
                       help='Use entity context extraction (KICGPTv2)')
    parser.add_argument('--use_relation_alignment', action="store_true",
                       help='Use relation self-alignment (KICGPTv2)')
    parser.add_argument('--use_reconstruction', action="store_true",
                       help='Use retrieval-augmented reconstruction (KICGPTv2)')
    parser.add_argument('--similarity_beta', default=0.33, type=float,
                       help='Similarity threshold for reconstruction')
    
    parser.add_argument('--max_tokens', default=300, type=int, help='max-token')
    parser.add_argument('--prompt_path', default="./prompts/relation_prediction.json")
    parser.add_argument('--prompt_name', default="chat")
    parser.add_argument('--overwrite', action="store_true")
    parser.add_argument('--device', default=0, help='the gpu device')
    
    parser.add_argument('--api_key', default="", type=str)
    parser.add_argument('--demon_per_step', default=8, type=int)
    parser.add_argument('--eff_demon_step', default=8, type=int, help='Number of demonstrations (δ=8 as per Appendix B best performance)')
    parser.add_argument('--max_demon_step', default=10, type=int)
    parser.add_argument('--max_llm_input_tokens', default=3750, type=int)
    parser.add_argument('--num_process', default=1, type=int, help='the number of multi-process')

    args = parser.parse_args()
    args.output_path = './outputs/' + args.dataset + '/output_relation.txt'
    args.chat_log_path = './outputs/' + args.dataset + '/chat_relation.txt'
    print("Start querying the LLM for relation prediction.")
    return args


if __name__ == '__main__':
    args = parse_args()
    if not args.api_key.startswith("sk-"):
        with open(args.api_key, "r") as f:
            all_keys = f.readlines()
            all_keys = [line.strip('\n') for line in all_keys]
            assert len(all_keys) == args.num_process, (len(all_keys), args.num_process)
    
    test_triplet = []

    # Load test data for relation prediction
    with open("datasets/" + args.dataset + "/test_relation.txt", 'r') as load_f:
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
# python relation_prediction.py --dataset fb15k-237 --use_entity_context --use_reconstruction --api_key <key>
