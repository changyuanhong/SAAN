import torch
import torch.backends.cudnn as cudnn
import os

# import models and utils
from params import get_parameters
from utils.ops import make_folder


from data.datasets import load_fs_data, load_raw, load_sample

import data.constants as cte


from models.trainer_losgan import Trainer

from models.tester_losgan import Trainer_classifier



def main(config):

    # For fast training
    cudnn.benchmark = True

    # Create directories if not exist

    make_folder(config.results, config.model_save_path, config.version)

    make_folder(config.results, config.image_path, config.version)

    img_path = os.path.join(config.results, config.image_path, config.version)

    dataset_path = os.path.join(config.results, config.dataset_path, config.version)





    print('config data_loader and build logs folder...')

    # & 生成的伪样本的数目，低于300准确率相对下降
    num_gen = 200

    # & LOSGAN的训练
    if config.train:
        for k in range(config.num_classes):

            dataroot = dataset_path + '/raw_' + str(k) + '.csv'
            dataloader = load_fs_data(dataroot, config.batch_size)

            trainer = Trainer(dataloader, config)


            a=trainer.train(k)

            trainer.evaluate(k, config.iteration, num_sample = num_gen)
         

    test_loader = load_raw(config.dataset, cte.SQ, cte.SQ_VS[0:config.num_classes], 200, 1064 , batchsize = 4)

    # ================== Generated samples mixed with fs samples for FSL ================== #
    raw_loader = load_sample(dataset_path, config.num_classes, fs = config.batch_size, num = num_gen, batchsize = 4)

    if config.test:
    
        trainer = Trainer_classifier(raw_loader, test_loader, config)
        trainer.train_classifier()


        # vis = Encoder_res(testloader,config,pretrained_model=num)
        # vis.test()


        # -----------------------------------------------------------------------------------
        #   & TSNE for representation and classification
        # -----------------------------------------------------------------------------------

        # tsne = Tsne(img_path)
        
        # classify_root = img_path + '/classify.csv'
        # representation_root = img_path + '/representation.csv'

        # '''
        # # & valfea, fea_code : representation result, 128-dimensiona
        # # & features, classification : the out put of the classifier
        # '''

        # tsne.plot_tsne(classify_root, config.num_classes, 'classification', config.dataset)
        # tsne.plot_tsne(representation_root, config.num_classes, 'representation', config.dataset)
    return a
if __name__ == '__main__':
    config = get_parameters()
    b=main(config)

    print('================== End ==================\n')


