package config


type LLMServerConfig struct{
	Host string `env:"HOST"`
	Port string `env:"PORT"`
}