import numpy as np
from tensorflow.keras.models import load_model
from src.utils.metrics import calculate_metrics


def inverse_target(
    values,
    X_reference,
    scaler,
    target_index
):
    """
    Converte apenas a variável alvo da escala normalizada para a escala original.
    """

    # Copia o último timestep de cada sequência
    dummy = X_reference[:, -1, :].copy()

    # Substitui apenas a coluna da variável alvo
    dummy[:, target_index] = values.flatten()

    # Desfaz a normalização
    dummy = scaler.inverse_transform(dummy)

    # Retorna somente a coluna da variável alvo
    return dummy[:, target_index]


def evaluate_model(
    model_path,
    scaler,
    X_test,
    y_test,
    features,
    target
):
    """
    Avalia o modelo treinado.
    """

    # Carrega o modelo salvo
    model = load_model(model_path)

    # Faz previsões
    y_pred = model.predict(X_test)

    # Descobre em qual coluna está a variável alvo
    target_index = features.index(target)

    # Desnormaliza previsão
    y_pred_inverse = inverse_target(
        y_pred,
        X_test,
        scaler,
        target_index
    )

    # Desnormaliza valores reais
    y_test_inverse = inverse_target(
        y_test,
        X_test,
        scaler,
        target_index
    )

    # Calcula as métricas de avaliação
    metrics = calculate_metrics(y_test_inverse, y_pred_inverse)

    return (
        y_pred_inverse,
        y_test_inverse,
        metrics
    )