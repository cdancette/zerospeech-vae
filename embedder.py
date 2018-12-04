import torch
from model import VAE
from data import get_dataset
import h5features
from pathlib import Path
import argparse

"""
def embed(output_path):
    model.eval()
    test_loss = 0
    embeddings = []
    with torch.no_grad():
        for i, data, in enumerate(data_loader):
            data = data.to(device)
            mu, logvar = self.encode(x.view(-1, self.input_size))
        	z = self.reparameterize(mu, logvar)
        	embeddings.append(z)
            #recon_batch, mu, logvar = model(data)
            #test_loss += loss_function(recon_batch, data, mu, logvar).item()
  	embeddings = np.vstack(embeddings)
  	np.save(output_path, embeddings)

"""

if __name__=='__main__':

	parser = argparse.ArgumentParser(description='VAE MNIST Example')

	parser.add_argument('-f', '--features-path', required=True)
	parser.add_argument('-p', '--model-path', required=True)
	parser.add_argument('-s', '--embedding-size', required=True, type=int)
	parser.add_argument('-o', '--output-embeddings', required=True)
	parser.add_argument('--input-size', type=int, default=40)
	parser.add_argument('--csv', help="store csv", action="store_true")

	parser.add_argument('--batch-size', type=int, default=64, metavar='N',
	                    help='input batch size for training (default: 128)')
	
	parser.add_argument('--no-cuda', action='store_true', default=False,
	                    help='enables CUDA training')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
	                    help='random seed (default: 1)')
	parser.add_argument('--log-interval', type=int, default=10, metavar='N',
	                    help='how many batches to wait before logging training status')
	
	args = parser.parse_args()
	args.cuda = not args.no_cuda and torch.cuda.is_available()

	output_dir = Path(args.output_embeddings)

	torch.manual_seed(args.seed)

	device = torch.device("cuda" if args.cuda else "cpu")

	kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

	#dataset = get_dataset(args.features_path)
	#data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)

	if args.csv:
		output_dir = Path(args.output_embeddings)
		output_dir.mkdir(exist_ok=True)
	else:
		output_dir = Path(args.output_embeddings).parent.mkdir(parents=True, exist_ok=True)
		writer = h5features.Writer(args.output_embeddings)

	print("Starting embedding with input size %s" % args.input_size)
	model = VAE(input_size=args.input_size, num_components=args.embedding_size).to(device)
	model.load_state_dict(torch.load(args.model_path))
	model.eval()
	with torch.no_grad():
		data = h5features.reader.Reader(args.features_path).read()
		dict_features = data.dict_features()
		dict_labels = data.dict_labels()
		
		for file in dict_features:
			print("Processing {}".format(file))
			inputs = dict_features[file]
			inputs = torch.FloatTensor(inputs)
			inputs = inputs.to(device)
			mu, logvar = model.encode(inputs.view(-1, args.input_size))
			z = model.reparameterize(mu, logvar).cpu().numpy()
			if args.csv:	
				with open(output_dir / '{file}.fea'.format(file=file), "w") as f:
					for i, line in enumerate(z):
						f.write("{time} {feat}\n".format(time=dict_labels[file][i], feat=" ".join(str(l) for l in line)))
			else:
				data = h5features.Data(items=[file], labels=[dict_labels[file]], features=[z], check=True)
				writer.write(data, 'features', append=True)

	if not args.csv:
		writer.close()

	#embed(args.output_embeddings)

	#for epoch in range(1, args.epochs + 1):
	#	train(epoch)
	#	test(epoch)
	#	torch.save(model.state_dict(), "model-%s.pt" % epoch)
		#with torch.no_grad():
		#    sample = torch.randn(64, args.embedding_size).to(device)
		#    sample = model.decode(sample).cpu()
