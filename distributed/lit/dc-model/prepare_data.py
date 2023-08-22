import os
import deepchem as dc


os.environ['DEEPCHEM_DATA_DIR'] = os.path.join(os.getcwd(), 'data')

featurizer = dc.feat.GroverFeaturizer(dc.feat.CircularFingerprint())
tasks, datasets, transformers = dc.molnet.load_delaney(featurizer=featurizer, transformers=[])

tasks, datasets, transformers = dc.molnet.load_delaney(featurizer=dc.feat.DummyFeaturizer(), transformers=[])
train, test, valid = datasets
from deepchem.feat.vocabulary_builders import GroverAtomVocabularyBuilder, GroverBondVocabularyBuilder

atom_vocab = GroverAtomVocabularyBuilder()
atom_vocab.build(train)
atom_vocab.save('data/delaney_atom_vocab.json')

bond_vocab = GroverAtomVocabularyBuilder()
bond_vocab.build(train)
bond_vocab.save('data/delaney_bond_vocab.json')
