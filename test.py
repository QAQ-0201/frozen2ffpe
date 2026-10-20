import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util


if __name__ == '__main__':
    opt = TestOptions().parse()

    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    dataset = create_dataset(opt)
    dataset2 = create_dataset(opt)
    train_dataset = create_dataset(util.copyconf(opt, phase="train"))
    model = create_model(opt)

    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    for i, (data,data2) in enumerate(zip(dataset,dataset2)):
        if i == 0:
            model.data_dependent_initialize(data,data2)
            model.setup(opt)
            model.parallelize()
            if opt.eval:
                model.eval()
        if i >= opt.num_test:
            break
        model.set_input(data,data2)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, width=opt.display_winsize)
    webpage.save()
