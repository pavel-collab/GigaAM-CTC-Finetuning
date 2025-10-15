from torch.utils.data import DataLoader
from src.data.dataset import collate_fn
from tqdm import tqdm
from src.utils.utils import get_gigaam_logprobs, decode_indices
import pywer
import editdistance

class BeamSearchEvaluator:
    def __init__(self, model, beam_search_decoder, lm_model=None):
        self.model = model
        self.beam_search_decoder = beam_search_decoder
        self.lm_model = lm_model

    def set_beam_search_decoder(self, beam_search_decoder):
        self.beam_search_decoder = beam_search_decoder

    def set_lm_model(self, lm_model):
        self.lm_model = lm_model

    def evaluate(self, dataset, batch_size=8, num_workers=2):
        # dataloader = torch.utils.data.DataLoader(
        #     dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers
        # )
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=1
        )

        refs, hyps, best_hyps = [], [], []
        rescored_hyps = [] if self.lm_model else None
        n_bests = []

        for wav_batch, wav_lengths, texts in tqdm(dataloader):
            logprob_batch, encoded_len_batch = get_gigaam_logprobs(self.model, wav_batch, wav_lengths)
            beamsearch_result = self.beam_search_decoder(logprob_batch.cpu(), encoded_len_batch.cpu())

            for i, result in enumerate(beamsearch_result):
                ref = texts[i]
                refs.append(ref)

                best_hyp, best_rescored_hyp = self._process_nbest(result, ref)
                hyps.append(best_hyp[0])
                best_hyps.append(best_hyp[1])
                n_bests.append(best_hyp[2])

                if self.lm_model:
                    rescored_hyps.append(best_rescored_hyp)

        return self._compute_metrics(refs, hyps, best_hyps, rescored_hyps, n_bests)

    def _process_nbest(self, result, ref):
        best_distance = float('inf')
        best_hyp = None
        first_hyp = None
        best_rescored_hyp = None
        best_score = -float('inf')
        nbest_list = []

        for j, candidate in enumerate(result):
            curr_tokens = candidate.tokens
            curr_hyp = decode_indices(curr_tokens, self.model)
            nbest_list.append({'hyp': curr_hyp, 'score': candidate.score})

            if j == 0:
                first_hyp = curr_hyp

            distance = editdistance.eval(ref.split(), curr_hyp.split())
            if distance < best_distance:
                best_distance = distance
                best_hyp = curr_hyp

            if self.lm_model:
                score = self.lm_model.score(curr_hyp) / len(curr_hyp)
                if score > best_score:
                    best_score = score
                    best_rescored_hyp = curr_hyp

        return (first_hyp, best_hyp, nbest_list), best_rescored_hyp

    def _compute_metrics(self, refs, hyps, best_hyps, rescored_hyps, n_bests):
        wer = pywer.wer(refs, hyps)
        oracle_wer = pywer.wer(refs, best_hyps)

        output = {
            'wer': wer,
            'oracle_wer': oracle_wer,
            'references': refs,
            'hypotheses': hyps,
            'oracle_hypotheses': best_hyps,
            'n_bests': n_bests
        }

        if self.lm_model:
            rescored_wer = pywer.wer(refs, rescored_hyps)
            output.update({
                'rescored_wer': rescored_wer,
                'rescored_hypotheses': rescored_hyps,
            })

        return output
    
class EnhancedBeamSearchEvaluator(BeamSearchEvaluator):
    def __init__(self, model, beam_search_decoder, lm_model=None, transformer_lm=None):
        super().__init__(model, beam_search_decoder, lm_model)
        self.transformer_lm = transformer_lm
    
    def _process_nbest(self, result, ref):
        best_distance = float('inf')
        best_hyp = None
        first_hyp = None
        best_rescored_hyp = None
        best_transformer_hyp = None
        nbest_list = []
        
        # Собираем n-best кандидатов
        for j, candidate in enumerate(result):
            curr_tokens = candidate.tokens
            curr_hyp = decode_indices(curr_tokens, self.model)
            nbest_list.append({'hyp': curr_hyp, 'score': candidate.score})
            
            if j == 0:
                first_hyp = curr_hyp
            
            distance = editdistance.eval(ref.split(), curr_hyp.split())
            if distance < best_distance:
                best_distance = distance
                best_hyp = curr_hyp
        
        # KenLM переранжирование
        best_kenlm_hyp = None
        if self.lm_model:
            best_score = -float('inf')
            for candidate in nbest_list:
                score = self.lm_model.score(candidate['hyp']) / len(candidate['hyp'])
                if score > best_score:
                    best_score = score
                    best_kenlm_hyp = candidate['hyp']
        
        # Трансформер LM переранжирование
        best_transformer_hyp = None
        if self.transformer_lm:
            reranked = self.transformer_lm.rerank_nbest(nbest_list)
            best_transformer_hyp = reranked[0]['hyp'] if reranked else None
        
        return (first_hyp, best_hyp, nbest_list), best_kenlm_hyp, best_transformer_hyp
    
    def evaluate(self, dataset, batch_size=8, num_workers=2):
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers
        )

        refs, hyps, best_hyps = [], [], []
        kenlm_hyps, transformer_hyps = [], []
        n_bests = []

        for wav_batch, wav_lengths, texts in tqdm(dataloader):
            logprob_batch, encoded_len_batch = get_gigaam_logprobs(self.model, wav_batch, wav_lengths)
            beamsearch_result = self.beam_search_decoder(logprob_batch.cpu(), encoded_len_batch.cpu())

            for i, result in enumerate(beamsearch_result):
                ref = texts[i]
                refs.append(ref)

                best_hyp, best_kenlm, best_transformer = self._process_nbest(result, ref)
                hyps.append(best_hyp[0])
                best_hyps.append(best_hyp[1])
                n_bests.append(best_hyp[2])
                
                if self.lm_model:
                    kenlm_hyps.append(best_kenlm)
                if self.transformer_lm:
                    transformer_hyps.append(best_transformer)

        return self._compute_metrics(refs, hyps, best_hyps, kenlm_hyps, transformer_hyps, n_bests)
    
    def _compute_metrics(self, refs, hyps, best_hyps, kenlm_hyps, transformer_hyps, n_bests):
        wer = pywer.wer(refs, hyps)
        oracle_wer = pywer.wer(refs, best_hyps)

        output = {
            'wer': wer,
            'oracle_wer': oracle_wer,
            'references': refs,
            'hypotheses': hyps,
            'oracle_hypotheses': best_hyps,
            'n_bests': n_bests
        }

        if self.lm_model:
            kenlm_wer = pywer.wer(refs, kenlm_hyps)
            output.update({
                'kenlm_wer': kenlm_wer,
                'kenlm_hypotheses': kenlm_hyps,
            })

        if self.transformer_lm:
            transformer_wer = pywer.wer(refs, transformer_hyps)
            output.update({
                'transformer_wer': transformer_wer,
                'transformer_hypotheses': transformer_hyps,
            })

        return output
    
'''
How to use new enhanced evaluator:

transformer_lm = TransformerLMReranker("microsoft/DialoGPT-small")
enhanced_evaluator = EnhancedBeamSearchEvaluator(
    model=model, # GigaAM model
    beam_search_decoder=beam_search_decoder,
    lm_model=lm_model,
    transformer_lm=transformer_lm
)

res = evaluator.evaluate(dataset)
print('\nBeam search WER: ', res['wer'])
print('\nBeam search OracleWER: ', res['oracle_wer'])
'''