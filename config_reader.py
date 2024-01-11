from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """
    Configuration Settings

    This class represents the configuration settings for your application.

    Attributes:
        save_dir (str): The directory path where models will be saved.
        model_config (SettingsConfigDict): Configuration settings for models, loaded from an environment file.
    """
    save_dir: str
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')


config = Config()
config.model_config['protected_namespaces'] = ()
