import datetime
import os
import time

import hydra
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from graph4nlp.pytorch.modules.evaluation.accuracy import Accuracy
from graph4nlp.pytorch.modules.graph_construction import *
from graph4nlp.pytorch.modules.graph_embedding_initialization.embedding_construction import \
    WordEmbedding
from graph4nlp.pytorch.modules.graph_embedding_initialization.graph_embedding_initialization import \
    GraphEmbeddingInitialization
from graph4nlp.pytorch.modules.graph_embedding_learning import *
from graph4nlp.pytorch.modules.loss.general_loss import GeneralLoss
from graph4nlp.pytorch.modules.prediction.classification.graph_classification import \
    FeedForwardNN
from graph4nlp.pytorch.modules.utils import constants as Constants
from graph4nlp.pytorch.modules.utils.generic_utils import EarlyStopping
from graph4nlp.pytorch.modules.utils.logger import Logger
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from trec import TrecDataset


class TextClassifier(nn.Module):
    def __init__(self, vocab, config, device):
        super(TextClassifier, self).__init__()
        self.config = config
        self.vocab = vocab
        embedding_style = {'single_token_item': True if config.graph_type != 'ie' else False,
                           'emb_strategy': config.get('emb_strategy', 'w2v'),
                           'num_rnn_layers': 1,
                           'bert_model_name': config.get('bert_model_name', 'bert-base-uncased'),
                           'bert_lower_case': True
                           }

        assert not (config.graph_type in ('node_emb', 'node_emb_refined') and config.gnn == 'gat'), \
            'dynamic graph construction does not support GAT'

        use_edge_weight = False
        self.graph_topology = GraphEmbeddingInitialization(
            vocab.in_word_vocab,
            embedding_style=embedding_style,
            hidden_size=config.num_hidden,
            word_dropout=config.word_dropout,
            rnn_dropout=config.rnn_dropout,
            fix_word_emb=not config.no_fix_word_emb,
            fix_bert_emb=not config.get('no_fix_bert_emb', False))

        if 'w2v' in self.graph_topology.embedding_layer.word_emb_layers:
            self.word_emb = self.graph_topology.embedding_layer.word_emb_layers[
                'w2v'].word_emb_layer
        else:
            self.word_emb = WordEmbedding(
                self.vocab.in_word_vocab.embeddings.shape[0],
                self.vocab.in_word_vocab.embeddings.shape[1],
                pretrained_word_emb=self.vocab.in_word_vocab.embeddings,
                fix_emb=not config.no_fix_word_emb,
                device=device).word_emb_layer

        heads = [config.gat_num_heads] * \
            (config.gnn_num_layers - 1) + [config.gat_num_out_heads]
        print(f"np.array(heads).shape = {np.array(heads).shape}")
        print(f"heads = {heads}")
        print(f"heads[0] = {heads[0]}")
        self.gnn = GraphSAGE(config.gnn_num_layers,
                             config.num_hidden,
                             config.num_hidden,
                             config.num_hidden,
                             config.graphsage_aggreagte_type,
                             direction_option=config.gnn_direction_option,
                             feat_drop=config.gnn_dropout,
                             bias=True,
                             norm=None,
                             activation=F.relu,
                             use_edge_weight=use_edge_weight)
        self.clf = FeedForwardNN(2 * config.num_hidden
                                 if config.gnn_direction_option == 'bi_sep'
                                 else config.num_hidden,
                                 6,
                                 [config.num_hidden],
                                 graph_pool_type=config.graph_pooling,
                                 dim=config.num_hidden,
                                 use_linear_proj=config.max_pool_linear_proj)

        self.loss = GeneralLoss('CrossEntropy')

    def forward(self, graph_list, tgt=None, require_loss=True):
        # build graph topology
        batch_gd = self.graph_topology(graph_list)

        # run GNN encoder
        self.gnn(batch_gd)

        # run graph classifier
        self.clf(batch_gd)
        logits = batch_gd.graph_attributes['logits']

        if require_loss:
            loss = self.loss(logits, tgt)
            return logits, loss
        else:
            return logits


class ModelHandler:
    def __init__(self, config):
        super(ModelHandler, self).__init__()
        self.config = config
        self.logger = Logger(self.config.out_dir, config={
                             k: v for k, v in self.config.items() if k != 'device'}, overwrite=True)
        self.logger.write(self.config.out_dir)
        self._build_device()
        self._build_dataloader()
        self._build_model()
        self._build_optimizer()
        self._build_evaluation()

    def _build_device(self):
        if not self.config.no_cuda and torch.cuda.is_available():
            print('[ Using CUDA ]')
            self.device = torch.device(
                'cuda' if self.config.gpu < 0 else 'cuda:%d' % self.config.gpu)
            torch.cuda.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            torch.backends.cudnn.deterministic = True
            cudnn.benchmark = False
        else:
            self.device = torch.device('cpu')

    def _build_dataloader(self):
        dynamic_init_topology_builder = None
        if self.config.graph_type == 'dependency':
            topology_builder = DependencyBasedGraphConstruction
            graph_type = 'static'
            merge_strategy = 'tailhead'
        elif self.config.graph_type == 'constituency':
            topology_builder = ConstituencyBasedGraphConstruction
            graph_type = 'static'
            merge_strategy = 'tailhead'
        elif self.config.graph_type == 'ie':
            topology_builder = IEBasedGraphConstruction
            graph_type = 'static'
            merge_strategy = 'global'
        elif self.config.graph_type == 'node_emb':
            topology_builder = NodeEmbeddingBasedGraphConstruction
            graph_type = 'dynamic'
            merge_strategy = None
        elif self.config.graph_type == 'node_emb_refined':
            topology_builder = NodeEmbeddingBasedRefinedGraphConstruction
            graph_type = 'dynamic'
            merge_strategy = 'tailhead'

            if self.config.init_graph_type == 'line':
                dynamic_init_topology_builder = None
            elif self.config.init_graph_type == 'dependency':
                dynamic_init_topology_builder = DependencyBasedGraphConstruction
            elif self.config.init_graph_type == 'constituency':
                dynamic_init_topology_builder = ConstituencyBasedGraphConstruction
            elif self.config.init_graph_type == 'ie':
                merge_strategy = 'global'
                dynamic_init_topology_builder = IEBasedGraphConstruction
            else:
                raise RuntimeError(
                    'Define your own dynamic_init_topology_builder')
        else:
            raise RuntimeError('Unknown graph_type: {}'.format(
                self.config.graph_type))

        topology_subdir = '{}_graph'.format(self.config.graph_type)
        if self.config.graph_type == 'node_emb_refined':
            topology_subdir += '_{}'.format(self.config.init_graph_type)

        dataset = TrecDataset(root_dir=self.config.get('root_dir', self.config.root_data_dir),
                              pretrained_word_emb_name=self.config.get(
                                  'pretrained_word_emb_name', "840B"),
                              merge_strategy=merge_strategy,
                              graph_name="dependency",
                              seed=self.config.seed,
                              thread_number=4,
                              port=9000,
                              timeout=15000,
                              word_emb_size=300,
                              topology_builder=topology_builder,
                              topology_subdir=topology_subdir,
                              dynamic_graph_type=self.config.graph_type if
                              self.config.graph_type in ('node_emb', 'node_emb_refined') else None,
                              dynamic_init_topology_builder=dynamic_init_topology_builder,
                              dynamic_init_topology_aux_args={'dummy_param': 0}, edge_strategy="homogeneous")
        self.dataset = dataset
        self.train_dataloader = DataLoader(dataset.train, batch_size=self.config.batch_size, shuffle=True,
                                           num_workers=self.config.num_workers,
                                           collate_fn=dataset.collate_fn)
        if hasattr(dataset, 'val') == False:
            dataset.val = dataset.test
        self.val_dataloader = DataLoader(dataset.val, batch_size=self.config.batch_size, shuffle=False,
                                         num_workers=self.config.num_workers,
                                         collate_fn=dataset.collate_fn)
        self.test_dataloader = DataLoader(dataset.test, batch_size=self.config.batch_size, shuffle=False,
                                          num_workers=self.config.num_workers,
                                          collate_fn=dataset.collate_fn)
        self.vocab = dataset.vocab_model
        # self.config.num_classes = dataset.num_classes
        self.num_train = len(dataset.train)
        self.num_val = len(dataset.val)
        self.num_test = len(dataset.test)
        print('Train size: {}, Val size: {}, Test size: {}'
              .format(self.num_train, self.num_val, self.num_test))
        self.logger.write('Train size: {}, Val size: {}, Test size: {}'
                          .format(self.num_train, self.num_val, self.num_test))

    def _build_model(self):
        self.model = TextClassifier(
            self.vocab, self.config, device=self.device).to(self.device)

    def _build_optimizer(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(parameters, lr=self.config.lr)
        self.stopper = EarlyStopping(os.path.join(
            self.config.out_dir, Constants._SAVED_WEIGHTS_FILE), patience=self.config.patience)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=self.config.lr_reduce_factor,
                                           patience=self.config.lr_patience, verbose=True)

    def _build_evaluation(self):
        self.metric = Accuracy(['accuracy'])

    def train(self):
        dur = []
        for epoch in range(self.config.epochs):
            self.model.train()
            train_loss = []
            train_acc = []
            t0 = time.time()
            for i, data in enumerate(self.train_dataloader):
                tgt = data['tgt_tensor'].to(self.device)
                data['graph_data'] = data['graph_data'].to(
                    self.device)
                logits, loss = self.model(
                    data['graph_data'], tgt, require_loss=True)

                # add graph regularization loss if available
                if data['graph_data'].graph_attributes.get('graph_reg', None) is not None:
                    loss = loss + \
                        data['graph_data'].graph_attributes['graph_reg']

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())

                pred = torch.max(logits, dim=-1)[1].cpu()
                train_acc.append(self.metric.calculate_scores(
                    ground_truth=tgt.cpu(), predict=pred.cpu(), zero_division=0)[0])
                dur.append(time.time() - t0)

            val_acc = self.evaluate(self.val_dataloader)
            self.scheduler.step(val_acc)
            print('Epoch: [{} / {}] | Time: {:.2f}s | Loss: {:.4f} | Train Acc: {:.4f} | Val Acc: {:.4f}'.
                  format(epoch + 1, self.config.epochs, np.mean(dur), np.mean(train_loss), np.mean(train_acc), val_acc))
            self.logger.write('Epoch: [{} / {}] | Time: {:.2f}s | Loss: {:.4f} | Train Acc: {:.4f} | Val Acc: {:.4f}'.
                              format(epoch + 1, self.config.epochs, np.mean(dur), np.mean(train_loss), np.mean(train_acc), val_acc))

            if self.stopper.step(val_acc, self.model):
                break

        return self.stopper.best_score

    def evaluate(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            pred_collect = []
            gt_collect = []
            for i, data in enumerate(dataloader):
                tgt = data['tgt_tensor'].to(self.device)
                data['graph_data'] = data['graph_data'].to(
                    self.device)
                logits = self.model(data['graph_data'], require_loss=False)
                pred_collect.append(logits)
                gt_collect.append(tgt)

            pred_collect = torch.max(
                torch.cat(pred_collect, 0), dim=-1)[1].cpu()
            gt_collect = torch.cat(gt_collect, 0).cpu()
            score = self.metric.calculate_scores(
                ground_truth=gt_collect, predict=pred_collect, zero_division=0)[0]

            return score

    def test(self):
        # restored best saved model
        # self.stopper.load_checkpoint(self.model)

        t0 = time.time()
        acc = self.evaluate(self.test_dataloader)
        dur = time.time() - t0
        print('Test examples: {} | Time: {:.2f}s |  Test Acc: {:.4f}'.
              format(self.num_test, dur, acc))
        self.logger.write('Test examples: {} | Time: {:.2f}s |  Test Acc: {:.4f}'.
                          format(self.num_test, dur, acc))

        return acc


@hydra.main(version_base=None, config_path="config/trec", config_name="config")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))

    # config = OmegaConf.to_yaml(cfg)
    print(config)
    OmegaConf.set_struct(config, False)
    # run model
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    ts = datetime.datetime.now().timestamp()
    config.out_dir += '_{}'.format(ts)
    print('\n' + config.out_dir)

    runner = ModelHandler(config)
    runner.dataset.train[10].show_graph(r"C:\Users\FPCC\Desktop\graphNN\figure.html")
    breakpoint()
    t0 = time.time()

    val_acc = runner.train()
    test_acc = runner.test()

    runtime = time.time() - t0
    print('Total runtime: {:.2f}s'.format(runtime))
    runner.logger.write('Total runtime: {:.2f}s\n'.format(runtime))
    runner.logger.close()

    print('val acc: {}, test acc: {}'.format(val_acc, test_acc))


if __name__ == '__main__':
    main()
