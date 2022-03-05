
from src.training.trainer import Trainer
import os
from src.models.mlp import MLP
from src.models.cnn import CNN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow import keras
import argparse

def parse_arguments():
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--model', type=str, required=True)
    my_parser.add_argument('--task', type=str, required=True)
    my_parser.add_argument('--tune', action='store_true')
    
    return my_parser.parse_args()

def main(model_type, tune, task):
    
    if task == "train" :
        pixels = 28*28
        num_categories = 10

        if model_type == 'mlp':
            model = MLP(pixels,num_categories)

        elif model_type == 'cnn':
            model = CNN(pixels,num_categories)
            
        
        if tune == False:
            model_object = model.baseline_model()

            trainer = Trainer(model_type)

            trainer.train(model_object)
            
            model_object.save('baseline_model_'+model_type+'.h5')


        elif tune == True:
            model_builder = model.model_builder

            trainer = Trainer(model_type)

            model_object = trainer.tune(model_builder, model_type)

            trainer.train(model_object)

            model_object.save('best_model_'+model_type+'.h5')
        
        trainer.plot_metrics()

        trainer.eval(model_object)

        predictions = trainer.predict(model_object)
        
        trainer.predict_at_random(predictions)
        
        trainer.confusion_matrix(predictions)

        trainer.class_report(predictions)
    
    elif task == 'test':
        if tune ==False:
            trained_model = keras.models.load_model('baseline_model_'+model_type+'.h5')
            
        elif tune ==True:
            trained_model = keras.models.load_model('best_model_'+model_type+'.h5')


        tester = Trainer(model_type)
        predictions = tester.predict(trained_model)
        tester.confusion_matrix(predictions)
        tester.class_report(predictions)

                
        

if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments.model, arguments.tune, arguments.task)
