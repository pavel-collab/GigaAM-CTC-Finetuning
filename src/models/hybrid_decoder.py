class HybridCTCDecoder:
    def __init__(self, ctc_decoder, transformer_lm, lm_weight=0.3):
        self.ctc_decoder = ctc_decoder
        self.transformer_lm = transformer_lm
        self.lm_weight = lm_weight
    
    def decode(self, logprobs, lengths):
        # Получаем n-best от базового CTC декодера
        results = self.ctc_decoder(logprobs, lengths)
        
        enhanced_results = []
        for result in results:
            enhanced_candidates = []
            for candidate in result:
                # Добавляем трансформер LM score к каждому кандидату
                lm_score = self.transformer_lm.score_sequence(candidate[0])
                enhanced_score = candidate.score + self.lm_weight * lm_score
                
                enhanced_candidates.append({
                    'text': candidate[0],
                    'tokens': candidate.tokens,
                    'score': enhanced_score,
                    'ctc_score': candidate.score,
                    'lm_score': lm_score
                })
            
            # Сортируем по enhanced score
            enhanced_candidates.sort(key=lambda x: x['score'], reverse=True)
            enhanced_results.append(enhanced_candidates)
        
        return enhanced_results

'''
How to use hybrid decoder:

hybrid_decoder = HybridCTCDecoder(beam_search_decoder, transformer_lm, lm_weight=0.3)

evaluator = BeamSearchEvaluator(model, hybrid_decoder, lm_model=lm_model)

res = evaluator.evaluate(custom_dataset)
print('\nBeam search WER: ', res['wer'])
print('\nBeam search OracleWER: ', res['oracle_wer'])
'''