
# coding: utf-8

import Sequence_Generator

sequence_generator = Sequence_Generator.SequenceGenerator()
sequence_generator.load_pretrained_model()

sampled_sequence = sequence_generator.sample_sequence("this", 20)
print(sampled_sequence)

