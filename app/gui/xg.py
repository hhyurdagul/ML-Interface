import json
import os
from tkinter import ttk, filedialog

from .backend import (
    handle_errors,
    DataHandler,
    ScalerHandler,
    LookbackHandler,
    ModelHandler,
    ForecastHandler,
)
from .components import (
    InputListComponent,
    ModelValidationComponent,
    CustomizeTrainSetComponent,
    ModelComponent,
    TestModelComponent,
)
from .helpers import popupmsg


class XGB:
    def __init__(self):
        self.root = ttk.Frame()
        self.data_handler = DataHandler()
        self.scaler_handler = ScalerHandler()
        self.lookback_handler = LookbackHandler()
        self.model_handler = ModelHandler(
            self.data_handler, self.scaler_handler, self.lookback_handler
        )
        self.forecast_handler = ForecastHandler(self.model_handler)

        self.input_list_component = InputListComponent(self.root, self.data_handler)
        self.model_validation_component = ModelValidationComponent(self.root)
        self.customize_train_set_component = CustomizeTrainSetComponent(
            self.root, self.scaler_handler, self.lookback_handler
        )

        self.model_component = ModelComponent(
            self.root,
            self.model_handler,
            self.create_model,
            self.save_model,
            self.load_model,
        )

        self.test_model_component = TestModelComponent(
            self.root, self.forecast_handler, self.forecast
        )

    def save_model(self):
        path = filedialog.asksaveasfilename()
        if not path:
            return

        save_params = {}
        save_params.update(self.input_list_component.get_params())
        save_params.update(self.model_validation_component.get_params())
        save_params.update(self.customize_train_set_component.get_params())
        save_params.update(self.model_component.get_params())

        os.mkdir(path)
        self.customize_train_set_component.save_files(path)
        self.model_component.save_files(path)

        with open(path + "/model.json", "w") as outfile:
            json.dump(save_params, outfile)

    def load_model(self):
        path = filedialog.askdirectory()
        if not path:
            return
        if not os.path.exists(path + "/model.json"):
            popupmsg("Model file not found")
            return
        with open(path + "/model.json") as infile:
            params = json.load(infile)

        self.customize_train_set_component.load_files(path)
        if not self.model_component.load_files(path):
            return

        self.input_list_component.set_params(params)
        self.model_validation_component.set_params(params)
        self.customize_train_set_component.set_params(params)
        self.model_component.set_params(params)

    def __check_errors(self):
        return handle_errors(
            self.input_list_component.check_errors,
            self.model_validation_component.check_errors,
            self.customize_train_set_component.check_errors,
            self.model_component.check_errors,
        )

    def __set_handlers(self, forecast: bool = False):
        if forecast:
            self.data_handler.set_variables(
                self.model_validation_component.validation_option.get(),
                self.model_validation_component.do_forecast_option.get(),
                self.model_validation_component.random_percent_var.get(),
                self.model_validation_component.cross_val_var.get(),
            )
        else:
            self.data_handler.set_input_variables(
                self.input_list_component.get_predictor_names(),
                self.input_list_component.get_target_name(),
            )
            self.data_handler.set_variables(
                self.model_validation_component.validation_option.get(),
                self.model_validation_component.do_forecast_option.get(),
                self.model_validation_component.random_percent_var.get(),
                self.model_validation_component.cross_val_var.get(),
            )

        self.scaler_handler.set_scalers(
            self.customize_train_set_component.scale_var.get()
        )

        self.customize_train_set_component.calculate_sliding()
        self.lookback_handler.set_variables(
            self.customize_train_set_component.lookback_val_var.get(),
            self.customize_train_set_component.seasonal_period_var.get(),
            self.customize_train_set_component.seasonal_val_var.get(),
            self.customize_train_set_component.sliding,
        )

        self.model_handler.set_variables(
            self.model_component.get_model_params(),
            self.model_component.get_grid_params(),
            self.model_component.grid_option_var.get(),
        )

    def create_model(self):
        if not self.__check_errors():
            return

        self.__set_handlers(False)
        self.model_handler.create_model()
        if not self.model_handler.model_created:
            return popupmsg("Model is not created!")

        if not self.data_handler.do_forecast:
            self.pred = self.model_handler.pred
            self.y_test = self.model_handler.y_test
            for i, j in enumerate(self.model_handler.loss):
                self.test_model_component.test_metrics_vars[i].set(j)

        if self.model_handler.grid_option:
            popupmsg("Best Params: " + str(self.model_handler.best_params))

    def forecast(self):
        if not self.model_handler.model_created:
            return popupmsg("Model is not created!")
        self.__set_handlers(True)
        self.test_model_component.forecast()
