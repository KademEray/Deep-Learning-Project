# model_training.py
import os
import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from sklearn.preprocessing import MinMaxScaler
from gymnasium import spaces
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import optuna
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
import shutil
from tqdm import tqdm
import time

# Verwende CUDA, wenn verfügbar
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TradingEnv(gym.Env):
    def __init__(self, df, risk_percentage=0.02):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.action_space = spaces.Discrete(3)  # Aktionen: 0 = Halten, 1 = Kaufen, 2 = Verkaufen
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(df.shape[1],), dtype=np.float32)
        self.current_step = 0 if len(self.df) > 0 else -1
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.position = 0  # Anzahl der gehaltenen Anteile
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.entry_price = 0
        self.trades = []
        self.risk_percentage = risk_percentage  # Maximaler Risikoanteil pro Trade

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.entry_price = 0
        self.trades = []

        return self._next_observation(), {}

    def _next_observation(self):
        if self.current_step < 0 or self.current_step >= len(self.df):
            return np.zeros(self.observation_space.shape,
                            dtype=np.float32)  # Leerer Beobachtungsraum bei ungültigem Schritt
        obs = np.array(self.df.iloc[self.current_step], dtype=np.float32)
        return obs

    def _take_action(self, action):
        if self.current_step < 0 or self.current_step >= len(self.df):
            return True  # Ende der Episode aufgrund eines ungültigen Schritts

        current_price = self.df['Close'].iloc[self.current_step]
        done = False

        # Berechnung des maximalen Einsatzes basierend auf Risikomanagement
        max_trade_size = self.balance * self.risk_percentage

        if action == 1:  # Kaufen
            if self.balance >= current_price:
                trade_size = max_trade_size // current_price
                if trade_size > 0:
                    self.position += trade_size
                    self.balance -= trade_size * current_price
                    self.entry_price = current_price
                    self.trades.append({'step': self.current_step, 'type': 'buy', 'price': current_price})

        elif action == 2:  # Verkaufen
            if self.position > 0:
                self.balance += self.position * current_price
                self.position = 0
                self.trades.append({'step': self.current_step, 'type': 'sell', 'price': current_price})

        if self.position > 0:
            if current_price <= self.entry_price * 0.98:  # Stop-Loss bei 2% Verlust
                self.balance += self.position * current_price
                self.position = 0
                self.trades.append({'step': self.current_step, 'type': 'stop_loss', 'price': current_price})

            elif current_price >= self.entry_price * 1.05:  # Take-Profit bei 5% Gewinn
                self.balance += self.position * current_price
                self.position = 0
                self.trades.append({'step': self.current_step, 'type': 'take_profit', 'price': current_price})

        self.net_worth = self.balance + self.position * current_price
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.net_worth <= self.initial_balance * 0.9:
            done = True

        return done

    def step(self, action):
        done = self._take_action(action)
        self.current_step += 1

        if self.current_step >= len(self.df) - 1:
            done = True

        reward = self.net_worth - self.initial_balance
        obs = self._next_observation() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {}

        return obs, reward, done, False, info


class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(
            *args,
            **kwargs,
            net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
            activation_fn=torch.nn.ReLU,
            ortho_init=False,
        )


def evaluate_model(model, env, n_eval_episodes=10):
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, return_episode_rewards=False)
    return mean_reward


def train_model(train_data, model_params, total_timesteps, model_path, early_stopping=False):
    # Ordner für das Modell erstellen, falls nicht vorhanden
    model_dir = os.path.dirname(model_path)
    os.makedirs(model_dir, exist_ok=True)

    # Überprüfen, ob das Modell bereits existiert
    if os.path.exists(model_path):
        user_input = input(f"Das Modell {model_path} existiert bereits. Möchten Sie es neu trainieren? (ja/nein): ")
        if user_input.lower() != 'ja':
            print(f"Verwende vorhandenes Modell: {model_path}")
            return PPO.load(model_path, device=device)

    env = SubprocVecEnv([lambda: TradingEnv(train_data) for _ in range(os.cpu_count())])
    model = PPO(CustomPolicy, env, verbose=1, device=device, **model_params)
    if early_stopping:
        callback = EvalCallback(env, best_model_save_path=model_path, log_path=model_path, eval_freq=5000, verbose=1)
        model.learn(total_timesteps=total_timesteps, callback=callback)
    else:
        with tqdm(total=total_timesteps, desc="Trainiere Modell", unit="steps") as pbar:
            def progress_callback(_locals, _globals):
                pbar.update(1)
                return True

            model.learn(total_timesteps=total_timesteps, callback=progress_callback)

    # Speichern des Modells nur, wenn es erfolgreich trainiert wurde
    if model:
        model.save(model_path)
    return model


if __name__ == "__main__":
    # Lade gesäuberte und erweiterte Daten
    data_directories = [
        "../data/processed_data/with_indicators",
        "../data/processed_data/without_indicators"
    ]
    data_files = []
    for directory in data_directories:
        data_files.extend([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')])

    if not data_files:
        raise ValueError(
            "Keine Daten zum Trainieren gefunden. Bitte sicherstellen, dass die Daten korrekt vorbereitet wurden.")
    combined_df = pd.concat([pd.read_csv(file, index_col=0) for file in data_files], ignore_index=True)

    # Entferne Zeilen mit fehlenden Werten
    combined_df = combined_df.dropna()

    # Aufteilen der Daten in Trainings- und Testdaten
    split_ratio = 0.8
    split_index = int(len(combined_df) * split_ratio)
    train_df_full = combined_df.iloc[:split_index].reset_index(drop=True)
    test_df = combined_df.iloc[split_index:].reset_index(drop=True)

    # Verzeichnisse für Modelle erstellen
    model_directory = "../models/trained"
    backup_directory = "../models/backup"
    os.makedirs(model_directory, exist_ok=True)
    os.makedirs(backup_directory, exist_ok=True)

    # Trainiere das Basis-Modell mit 100% der Daten
    model_params = {
        'learning_rate': 0.03,
        'n_steps': 4096,
        'batch_size': 64,
        'gae_lambda': 0.95,
        'gamma': 0.99,
        'ent_coef': 0.0,
        'clip_range': 0.2,
        'n_epochs': 10,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
    }

    model_100_path = os.path.join(model_directory, "ppo_trading_model_100.zip")
    model_100 = train_model(train_df_full, model_params, total_timesteps=50000, model_path=model_100_path)

    # Trainiere Modelle mit verschiedenen Einstellungen basierend auf dem Basis-Modell
    risk_percentages = [0.01, 0.02, 0.05]
    for risk in risk_percentages:
        model_risk_path = os.path.join(model_directory, f"ppo_trading_model_risk_{int(risk * 100)}.zip")
        train_env = TradingEnv(train_df_full, risk_percentage=risk)
        model = PPO.load(model_100_path, env=train_env, device=device)
        model.learn(total_timesteps=50000)
        model.save(model_risk_path)

    # Vergleich verschiedener Optimierungsalgorithmen
    optimizers = ['adam', 'sgd']
    for optimizer in optimizers:
        model_optimizer_path = os.path.join(model_directory, f"ppo_trading_model_optimizer_{optimizer}.zip")
        model = PPO.load(model_100_path, env=SubprocVecEnv([lambda: TradingEnv(train_df_full)]), device=device)
        if optimizer == 'sgd':
            model.policy.optimizer = torch.optim.SGD(model.policy.parameters(), lr=0.03)
        else:
            model.policy.optimizer = torch.optim.Adam(model.policy.parameters(), lr=0.03)
        model.learn(total_timesteps=50000)
        model.save(model_optimizer_path)

    # Trainiere Modell ohne zusätzliche technische Indikatoren
    train_df_no_indicators = combined_df.iloc[:split_index].reset_index(drop=True)
    model_no_indicators_path = os.path.join(model_directory, "ppo_trading_model_no_indicators.zip")
    train_env_no_indicators = TradingEnv(train_df_no_indicators)
    model = PPO(CustomPolicy, train_env_no_indicators, verbose=1, device=device, **model_params)
    model.learn(total_timesteps=50000)
    model.save(model_no_indicators_path)

    # Backup aller Modelle
    for model_file in os.listdir(model_directory):
        shutil.copy(os.path.join(model_directory, model_file), backup_directory)

    print("Alle Modelle erfolgreich trainiert und gesichert.")