import wandb
import pprint

sweep_config = {"method": "random"}

metric = {
	"name": "LEVENSHTEIN",
	"goal": "minimize" 
}

sweep_config["metric"] = metric 

parameters_dict = {
	"batch_size":{
		"values": [4, 8, 16]
	},
	"epoch_num":{
		"values": [20, 25, 30]
	},
	"optimizer":{
		"values":["adam", "sgd"]
	},
	"n_res_cnn_layers":{
		"values": [3, 6, 12, 24]
	},
	"cnn_dropout":{
		"values":[0.2, 0.3, 0.4]
	},
	"n_rnn_layers":{
		"values": [5, 7, 9]
	},
	"rnn_dim": {
		"values": [128, 256, 512]
	},
	"kernel_size":{
		"values":[3, 5]
	},
	"kernel_stride":{
		"values":[1,2]
	}

}

sweep_config['parameters'] = parameters_dict
