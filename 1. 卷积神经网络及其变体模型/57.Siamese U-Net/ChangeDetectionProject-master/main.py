import matplotlib, os
if not('DISPLAY' in os.environ):
    matplotlib.use("Agg")

import DataLoader, DataPreprocesser, Dataset, Debugger, Settings, ModelHandler, Evaluator
from timeit import default_timer as timer
from datetime import *

months = ["unk","jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
month = (months[datetime.now().month])
day = str(datetime.now().day)

import argparse

parser = argparse.ArgumentParser(description='Project: Change detection on aerial images.')
parser.add_argument('-name', help='run name - will output in this dir', default="Run-"+month+"-"+day)
parser.add_argument('-KFOLDS', help='Number of folds', default='5')
parser.add_argument('-FOLD_I', help='This fold i', default='0')
parser.add_argument('-model_backend', help='Model used in the encoder part of the U-Net structures model', default='resnet50')
parser.add_argument('-train_epochs', help='How many epochs', default='100')
parser.add_argument('-train_batch', help='How big batch size', default='8')

def main(args):
    print(args)

    settings = Settings.Settings(args)

    # We already did these
    # ResNet50 and indices: 5, 2, 7, 3 (doing ? r.n.)
    settings.TestDataset_Fold_Index = int(args.FOLD_I) # can be 0 to 9 (K-1)
    settings.TestDataset_K_Folds = int(args.KFOLDS)
    assert settings.TestDataset_Fold_Index < settings.TestDataset_K_Folds
    kfold_txt = "KFold_"+str(settings.TestDataset_Fold_Index)+"z"+str(settings.TestDataset_K_Folds)
    print(kfold_txt)

    # resnet 101 approx 5-6 hours (per fold - might be a bit less ...)
    # resnet 50  approx 3-4 hours
    model_txt = "cleanManual_"+args.train_epochs+"ep_ImagenetWgenetW_"+args.model_backend+"-"+args.train_batch+"batch_Augmentation1to1_ClassWeights1to3_TestVal"
    print(model_txt)

    dataset = Dataset.Dataset(settings)
    evaluator = Evaluator.Evaluator(settings)

    #settings.run_name = settings.run_name + "AYRAN"
    show = False
    save = True

    #dataset.dataset
    settings.model_backend = args.model_backend
    settings.train_epochs = int(args.train_epochs)
    settings.train_batch = int(args.train_batch)
    model = ModelHandler.ModelHandler(settings, dataset)

    model.model.train(show=show,save=save)

    # Model 2 ...

    # TODO Note:
    # - change settings.run_name to have saved plots
    # write down:
    # - model bottom (resnet34 ?)
    # - initial weights (imagenet ?)
    # - used augmentation ?
    # - epoch number
    # - class weights changed ?
    # - ... any other special cool thing ...

    # K-Fold_Crossval:
    #model.model.save("/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/weightsModel2_"+model_txt+"_["+kfold_txt+"].h5")
    model.model.save("/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/weightsModel2_"+model_txt+"_["+kfold_txt+"].h5")


    # Next = train Resnet50 on the same dataset without the whole STRIP2 (to have some large Test images)

    #model.model.load("/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/weightsModel2_cleanManual_100ep_ImagenetWgenetW_seresnext50-8batch_Augmentation1to1_ClassWeights1to3.h5")

    #model.model.load("/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/weightsModel2_cleanManual-noStrip2_100ep_ImagenetWgenetW_resnet50-16batch_Augmentation1to1_ClassWeights1to3.h5")
    #model.model.load("/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/weightsModel2_cleanManual_100ep_ImagenetWgenetW_resnet101-8batch_Augmentation1to1_ClassWeights1to3.h5")

    # Senet154 crashed, 10hrs train + Imagenet weights + Data Aug 1:1 + Class weight 1:3
    #model.model.load("/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/weightsModel2_cleanManual_XYZep_ImagenetW_senet154-4batch_Augmentation1to1_ClassWeights1to3_early_stop_save_26mar-7am(cca10hrs).h5")

    # Seresnet34 + Imagenet weights + Data Aug 1:1 + Class weight 1:3
    #model.model.load("/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/weightsModel2_cleanManual_100ep_ImagenetWgenetW_seresnet34_Augmentation1to1_ClassWeights1to3.h5")

    # Resnet50 (batch 16) + Imagenet weights + Data Aug 1:1 + Class weight 1:3
    #model.model.load("/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/weightsModel2_cleanManual_100ep_ImagenetWgenetW_resnet50-16batch_Augmentation1to1_ClassWeights1to3.h5")

    # Resnet34 + Imagenet weights + Data Aug 1:1 + Class weight 1:3
    #model.model.load("/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/weightsModel2_cleanManual_100ep_ImagenetW_Resnet34_Augmentation1to1_ClassWeights1to3.h5")

    # Resnet34 + Imagenet weights + No Data Aug + Class weight 1:3
    #model.model.load("/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/weightsModel2_cleanManual_100ep_ImagenetBase.h5")
    #model.model.load("/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/weightsModel2_cleanManual_54ep_ImagenetBase_best_so_far_for_eastly_stops.h5") # early stop at 54 ep

    # Resnet34 + Custom DSM weights + No Data Aug + Class weight 1:3
    #model.model.load("/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/weightsModel2_cleanManual_100ep_CustomDSMBase.h5")
    #model.model.load("/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/weightsModel2_cleanManual_49ep_CustomDSMBase_best_so_far_for_eastly_stops.h5")


    #model.model.load("/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/weightsModel2_cleanManual_25ep_ImagenetFrozenEnc.h5") # 26,428,523 > 5,139,429 trainable params - faster?
    #model.model.load("/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/weightsModel2_cleanManual_.h5")
    # ...


    SAVE_ALL_FOLDER = model_txt+"PLOTS/"
    SAVE_ALL_PLOTS = SAVE_ALL_FOLDER+"plot"
    # DEBUG_SAVE_ALL_THR_PLOTS = None
    if not os.path.exists(SAVE_ALL_FOLDER):
        os.makedirs(SAVE_ALL_FOLDER)

    evaluator.unified_test_report([model.model.model], dataset.test, validation_set=dataset.val, postprocessor=model.model.dataPreprocesser,
                                                                               name=SAVE_ALL_PLOTS,
                                                                               optionally_save_missclassified=True)

    #model.model.test(evaluator,show=show,save=save)




    #model.model.test_on_specially_loaded_set(evaluator,show=show,save=save) # << LOAD just strip 2 here, use FCN to predict large areas
    #model.model.test_show_on_train_data_to_see_overfit(evaluator)

if __name__ == '__main__':
    args = parser.parse_args()

    start = timer()

    main(args)

    end = timer()
    time = (end - start)

    print("This run took "+str(time)+"s ("+str(time/60.0)+"min)")

    import keras
    keras.backend.clear_session()
