import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────
#  1. DATA LOADING & VALIDATION
# ─────────────────────────────────────────

def load_data(filepath: str = "Salary_Data.csv") -> pd.DataFrame:
    """Load and validate the dataset."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"[INFO] '{filepath}' not found. Generating synthetic dataset...")
        df = generate_synthetic_data()

    required_cols = {"YearsExperience", "Salary"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required_cols}")

    df.dropna(inplace=True)
    print(f"[INFO] Dataset loaded: {len(df)} records\n")
    print(df.describe().round(2))
    return df


def generate_synthetic_data(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic salary data for demo purposes."""
    np.random.seed(seed)
    years = np.round(np.random.uniform(0.5, 15, n), 1)
    salary = 30000 + (years * 7500) + np.random.normal(0, 8000, n)
    return pd.DataFrame({"YearsExperience": years, "Salary": salary.round(2)})


# ─────────────────────────────────────────
#  2. MODEL TRAINING
# ─────────────────────────────────────────

def train_model(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split data and train Linear Regression model."""
    X = df[["YearsExperience"]]
    y = df["Salary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test


# ─────────────────────────────────────────
#  3. EVALUATION
# ─────────────────────────────────────────

def evaluate_model(model, X_test, y_test) -> dict:
    """Compute and display evaluation metrics."""
    y_pred = model.predict(X_test)

    metrics = {
        "MSE":   mean_squared_error(y_test, y_pred),
        "RMSE":  np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE":   mean_absolute_error(y_test, y_pred),
        "R²":    r2_score(y_test, y_pred),
        "Coeff": model.coef_[0],
        "Intercept": model.intercept_,
    }

    print("\n" + "═" * 45)
    print("  MODEL EVALUATION REPORT")
    print("═" * 45)
    print(f"  Coefficient (slope) : ₹{metrics['Coeff']:>12,.2f}")
    print(f"  Intercept           : ₹{metrics['Intercept']:>12,.2f}")
    print("─" * 45)
    print(f"  Mean Squared Error  : {metrics['MSE']:>15,.2f}")
    print(f"  Root MSE (RMSE)     : ₹{metrics['RMSE']:>12,.2f}")
    print(f"  Mean Absolute Error : ₹{metrics['MAE']:>12,.2f}")
    print(f"  R² Score            : {metrics['R²']:>15.4f}  ({metrics['R²']*100:.1f}% variance explained)")
    print("═" * 45)

    return metrics, y_pred


# ─────────────────────────────────────────
#  4. PREDICTION UTILITY
# ─────────────────────────────────────────

def predict_salary(model, years: float) -> float:
    """Predict salary for a given years of experience."""
    pred = model.predict([[years]])[0]
    print(f"\n  Predicted Salary for {years} years experience: ₹{pred:,.2f}")
    return pred


# ─────────────────────────────────────────
#  5. VISUALIZATION
# ─────────────────────────────────────────

def plot_results(df, model, X_train, X_test, y_train, y_test, y_pred, metrics):
    """Generate a 2×2 dashboard of plots."""
    fig = plt.figure(figsize=(14, 10), facecolor="#0f1117")
    fig.suptitle("Salary Prediction — Linear Regression Dashboard",
                 fontsize=16, fontweight="bold", color="white", y=0.98)

    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)
    ax_colors = {"bg": "#1a1d27", "accent": "#4f8ef7", "green": "#2ecc71",
                 "red": "#e74c3c", "text": "#dde3f0", "grid": "#2a2d3e"}

    def style_ax(ax, title):
        ax.set_facecolor(ax_colors["bg"])
        ax.tick_params(colors=ax_colors["text"], labelsize=9)
        ax.xaxis.label.set_color(ax_colors["text"])
        ax.yaxis.label.set_color(ax_colors["text"])
        ax.title.set_color(ax_colors["text"])
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.grid(True, color=ax_colors["grid"], linestyle="--", alpha=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor(ax_colors["grid"])

    X_all = df[["YearsExperience"]]
    x_line = np.linspace(df["YearsExperience"].min(), df["YearsExperience"].max(), 200).reshape(-1, 1)

    # — Plot 1: Regression Line on Full Data —
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, "Regression Line — Full Dataset")
    ax1.scatter(df["YearsExperience"], df["Salary"], color=ax_colors["accent"], alpha=0.6, s=40, label="Actual")
    ax1.plot(x_line, model.predict(x_line), color=ax_colors["red"], lw=2.5, label="Regression Line")
    ax1.set_xlabel("Years of Experience")
    ax1.set_ylabel("Salary (₹)")
    ax1.legend(facecolor=ax_colors["bg"], labelcolor=ax_colors["text"], fontsize=8)

    # — Plot 2: Train vs Test split —
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, "Train / Test Split View")
    ax2.scatter(X_train, y_train, color=ax_colors["green"], alpha=0.7, s=40, label="Train")
    ax2.scatter(X_test, y_test, color=ax_colors["red"], alpha=0.8, s=50, marker="^", label="Test")
    ax2.plot(x_line, model.predict(x_line), color="white", lw=2, linestyle="--")
    ax2.set_xlabel("Years of Experience")
    ax2.set_ylabel("Salary (₹)")
    ax2.legend(facecolor=ax_colors["bg"], labelcolor=ax_colors["text"], fontsize=8)

    # — Plot 3: Actual vs Predicted —
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3, "Actual vs Predicted (Test Set)")
    ax3.scatter(y_test, y_pred, color=ax_colors["accent"], edgecolors="white", s=60, alpha=0.85)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax3.plot([min_val, max_val], [min_val, max_val], color=ax_colors["red"], lw=2, linestyle="--", label="Perfect Fit")
    ax3.set_xlabel("Actual Salary (₹)")
    ax3.set_ylabel("Predicted Salary (₹)")
    ax3.legend(facecolor=ax_colors["bg"], labelcolor=ax_colors["text"], fontsize=8)

    # — Plot 4: Residuals —
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4, "Residuals (Prediction Errors)")
    residuals = y_test.values - y_pred
    ax4.scatter(y_pred, residuals, color=ax_colors["accent"], alpha=0.75, s=50)
    ax4.axhline(0, color=ax_colors["red"], lw=2, linestyle="--")
    ax4.set_xlabel("Predicted Salary (₹)")
    ax4.set_ylabel("Residual (₹)")
    rmse_label = f"RMSE: ₹{metrics['RMSE']:,.0f}  |  R²: {metrics['R²']:.3f}"
    ax4.text(0.02, 0.95, rmse_label, transform=ax4.transAxes,
             color=ax_colors["green"], fontsize=8.5, va="top")

    plt.savefig("salary_dashboard.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
    print("\n[INFO] Dashboard saved as 'salary_dashboard.png'")
    plt.show()


# ─────────────────────────────────────────
#  6. MAIN PIPELINE
# ─────────────────────────────────────────

def main():
    print("\n" + "═" * 45)
    print("  SALARY PREDICTION MODEL  |  Mini Project")
    print("═" * 45)

    df = load_data("Salary_Data.csv")
    model, X_train, X_test, y_train, y_test = train_model(df)
    metrics, y_pred = evaluate_model(model, X_test, y_test)

    # Example predictions
    print("\n  SAMPLE PREDICTIONS:")
    for yrs in [1, 3, 5, 7, 10, 15]:
        predict_salary(model, yrs)

    plot_results(df, model, X_train, X_test, y_train, y_test, y_pred, metrics)


if __name__ == "__main__":
    main()