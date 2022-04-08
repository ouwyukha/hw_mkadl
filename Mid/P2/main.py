import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import argparse
import time, math, random
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import os
from torch import optim
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

import data as dataloader
import seq2seq

class Main:
    def __init__(self, embedding_size=300, hidden_size=300, rnnType='gru', teacher_focing_ratio=0.5):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.rnnType = rnnType
        self.teacher_focing_ratio = teacher_focing_ratio

    def prepareDataset(self, filepath='./data/q-a.txt', q='q', a='a'):
        self.input_lang, self.output_lang, self.pairs = dataloader.perpare_dataset(filepath, q, a)

    def generateEncoder(self):
        if not self.input_lang:
            print("Dataset hasn't been prepared yet, please run prepareDataset function!")
            return

        self.encoder = seq2seq.GRUEncoder(input_size=self.input_lang.num_words, embedding_size=self.embedding_size,
                                  hidden_size=self.hidden_size,type=self.rnnType).to(self.device)

    def generateDecoder(self):
        if not self.output_lang:
            print("Dataset hasn't been prepared yet, please run prepareDataset function!")
            return

        self.decoder = seq2seq.GRUDecoder(output_size=self.output_lang.num_words, embedding_size=self.embedding_size,
                                  hidden_size=self.hidden_size,type=self.rnnType).to(self.device)

    def loadEncoder(self):
        if os.path.isfile('./'+self.rnnType+'_encoder.pth'):
            self.encoder.load_state_dict(torch.load('encoder.pth'))
            print("Successfully loaded encoder state")
        else:
            print("Failed to find encoder state")

    def loadDecoder(self):
        if os.path.isfile('./'+self.rnnType+'_decoder.pth'):
            self.decoder.load_state_dict(torch.load('decoder.pth'))
            print("Successfully loaded decoder state")
        else:
            print("Failed to find decoder state")

    def asMinutes(self, s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def timeSince(self, since, percent):
        now = time.time()
        s = now - since
        es = s / (percent)
        rs = es - s
        return '%s (- %s)' % (self.asMinutes(s), self.asMinutes(rs))

    def showPlot(self, points):
        plt.figure()
        fig, ax = plt.subplots()
        loc = ticker.MultipleLocator(base=0.2)
        ax.yaxis.set_major_locator(loc)
        plt.plot(points)

    def indexesFromSentence(self, lang, sentence):
        return [lang.word2index[word] for word in sentence.split(' ')]

    def tensorFromSentences(self, lang, sentence):
        indexes = self.indexesFromSentence(lang, sentence)
        indexes.append(dataloader.EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=self.device).view(-1, 1)

    def tensorFromPair(self, pair):
        input_tensor = self.tensorFromSentences(self.input_lang, pair[0])
        target_tensor = self.tensorFromSentences(self.output_lang, pair[1])
        return (input_tensor, target_tensor)

    def train(self, input_tensor, target_tensor, encoder_optimizer, decoder_optimizer,
            criterion, max_length=dataloader.MAX_LENGTH):
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_length, target_length = input_tensor.size(0), target_tensor.size(0)

        if self.rnnType=='gru':
            encoder_hidden = self.encoder.init_hidden().to(self.device)
        else:
            encoder_hidden = (self.encoder.init_hidden().to(self.device), self.encoder.init_hidden().to(self.device))

        encoder_outputs = torch.zeros(max_length, self.encoder.hidden_size).to(self.device)

        loss = 0
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)

            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input=torch.tensor([[dataloader.SOS_token]]).to(self.device)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < self.teacher_focing_ratio else False

        if use_teacher_forcing:
            for di in range(target_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

                loss += criterion(decoder_output, target_tensor[di])

                decoder_input = target_tensor[di]
        else:
            for di in range(target_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

                topv, topi = decoder_output.topk(1)

                decoder_input = topi.squeeze().detach()
                loss += criterion(decoder_output, target_tensor[di])

                if decoder_input.item() == dataloader.EOS_token:
                    break

        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.item() / target_length

    def trainiters(self, n_iters=10, print_every=1000, plot_every=1000, learning_rate=.01):
        start = time.time()
        plot_losses = []
        print_loss_total, plot_loss_total = 0,0

        train_pairs=[self.tensorFromPair(random.choice(self.pairs)) for i in range(n_iters)]

        encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=learning_rate)

        criterion=nn.NLLLoss()

        for iter in range(1, n_iters+1):
            pair = train_pairs[iter-1]

            input_tensor, target_tensor = pair[0], pair[1]

            loss = self.train(input_tensor, target_tensor, encoder_optimizer, decoder_optimizer, criterion)

            print_loss_total += loss
            plot_loss_total += loss

            if iter %print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (self.timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg))
            if iter % plot_every:
                plot_loss_avg = plot_loss_total / print_every

                plot_losses.append(plot_loss_avg)
                plot_loss_total=0

        self.showPlot(plot_losses)

        if self.rnnType == 'gru':
            plt.savefig('baseline-GRU-loss')
        else:
            plt.savefig('baseline-LSTM-loss')

        torch.save(self.encoder.state_dict(), './'+self.rnnType+'_encoder.pth')
        torch.save(self.decoder.state_dict(), './'+self.rnnType+'_decoder.pth')

    def predict_sentence(self, pair, output):
        print('Question: {}\nAnswer: {}'.format(pair[0], pair[1]))
        print('Predict Answer:{}'.format(output), end='\n\n')

    def evaluate(self, sentence, max_length=dataloader.MAX_LENGTH):
        with torch.no_grad():
            input_tensor = self.tensorFromSentences(self.input_lang, sentence)

            input_length = input_tensor.size(0)

            if self.rnnType == 'gru':
                encoder_hidden = self.encoder.init_hidden().to(self.device)
            else:
                encoder_hidden = (self.encoder.init_hidden().to(self.device), self.encoder.init_hidden().to(self.device))

            encoder_outputs = torch.zeros(max_length, self.encoder.hidden_size).to(self.device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
                encoder_outputs[ei] = encoder_output[0, 0]

            decoder_input = torch.tensor([[dataloader.SOS_token]]).to(self.device)
            decoder_hidden = encoder_hidden

            decoded_words = []
            for di in range(max_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.data.topk(1)

                if topi.item() == dataloader.EOS_token:
                    break
                else:
                    decoded_words.append(self.output_lang.index2word[topi.item()])

                decoder_input = topi.squeeze().detach()

        return decoded_words

    def evaluateiters(self, pairs, random_seed=0):
        start = time.time()
        cc = SmoothingFunction()
        train_pairs, test_pairs = train_test_split(pairs, test_size=0.15, random_state=random_seed)

        scores = []
        for pi, pair in enumerate(test_pairs):
            output_words = self.evaluate(pair[0])
            output_sentence = ' '.join(output_words)

            self.predict_sentence(pair, output_sentence)

            ref = pair[1].split()
            hyp = output_words
            scores.append(sentence_bleu([ref], hyp, smoothing_function=cc.method3) * 100.)

        print('BLEU: {:.4}'.format(sum(scores) / len(test_pairs)))

if __name__ == "__main__":
    main = Main(embedding_size=300, hidden_size=300, rnnType='gru', teacher_focing_ratio=0.5)

    main.prepareDataset(filepath='./data/q-a.txt', q='q', a='a')

    main.generateEncoder()
    main.generateDecoder()

    main.loadEncoder()
    main.loadDecoder()

    print(main.encoder, main.decoder)

    main.trainiters(n_iters=10, print_every=1000, plot_every=1000, learning_rate=.0)

    main.evaluateiters(main.pairs, random_seed=0)

