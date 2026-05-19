"use client";

import { useState } from "react";
import axios from "axios";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface FormData {
  gender: string;
  age: number | "";
  customer_segment: string;
  tenure_months: number | "";
  signup_channel: string;
  contract_type: string;
  monthly_logins: number | "";
  weekly_active_days: number | "";
  avg_session_time: number | "";
  features_used: number | "";
  usage_growth_rate: number | "";
  last_login_days_ago: number | "";
  monthly_fee: number | "";
  total_revenue: number | "";
  payment_method: string;
  payment_failures: number | "";
  discount_applied: string;
  price_increase_last_3m: string;
  support_tickets: number | "";
  avg_resolution_time: number | "";
  complaint_type: string;
  csat_score: number | "";
  escalations: number | "";
  email_open_rate: number | "";
  marketing_click_rate: number | "";
  nps_score: number | "";
  survey_response: string;
  referral_count: number | "";
}

interface PredictionResult {
  churn: number;
  probability: number;
  threshold: number;
  risk_level: "low" | "medium" | "high";
}

const steps = [
  {
    title: "Informations Personnelles",
    fields: ["gender", "age", "customer_segment"],
  },
  {
    title: "Engagement",
    fields: [
      "tenure_months",
      "signup_channel",
      "contract_type",
      "monthly_logins",
      "weekly_active_days",
    ],
  },
  {
    title: "Utilisation",
    fields: [
      "avg_session_time",
      "features_used",
      "usage_growth_rate",
      "last_login_days_ago",
    ],
  },
  {
    title: "Facturation",
    fields: [
      "monthly_fee",
      "total_revenue",
      "payment_method",
      "payment_failures",
      "discount_applied",
      "price_increase_last_3m",
    ],
  },
  {
    title: "Support",
    fields: [
      "support_tickets",
      "avg_resolution_time",
      "complaint_type",
      "csat_score",
      "escalations",
    ],
  },
  {
    title: "Communication & Satisfaction",
    fields: [
      "email_open_rate",
      "marketing_click_rate",
      "nps_score",
      "survey_response",
      "referral_count",
    ],
  },
];

const fieldLabels: Record<string, string> = {
  gender: "Genre",
  age: "Âge",
  customer_segment: "Type de client (ex: PME, Enterprise)",
  tenure_months: "Ancienneté (en mois)",
  signup_channel: "Canal d'inscription (ex: Web, Mobile, Email)",
  contract_type: "Type de contrat",
  monthly_logins: "Connexions par mois",
  weekly_active_days: "Jours actifs par semaine",
  avg_session_time: "Durée moyenne de session (minutes)",
  features_used: "Nombre de fonctionnalités utilisées (ex: 5, 10, 15)",
  usage_growth_rate: "Croissance d'utilisation (%)",
  last_login_days_ago: "Jours depuis dernière connexion",
  monthly_fee: "Montant d'abonnement mensuel",
  total_revenue: "Revenu total généré",
  payment_method: "Méthode de paiement",
  payment_failures: "Nombre d'échecs de paiement",
  discount_applied: "Remise appliquée",
  price_increase_last_3m: "Augmentation de prix (3 derniers mois)",
  support_tickets: "Nombre de demandes de support",
  avg_resolution_time: "Temps de résolution moyen (heures)",
  complaint_type: "Type de réclamation",
  csat_score: "Score de satisfaction (0-5)",
  escalations: "Problèmes escaladés",
  email_open_rate: "Taux d'ouverture des emails (%)",
  marketing_click_rate: "Taux de clic marketing (%)",
  nps_score: "Indice de recommandation (-100 à 100)",
  survey_response: "Réponse à l'enquête (ex: Satisfied)",
  referral_count: "Clients recommandés",
};

const fieldTypes: Record<string, string> = {
  gender: "select",
  age: "number",
  customer_segment: "text",
  tenure_months: "number",
  signup_channel: "text",
  contract_type: "select",
  monthly_logins: "number",
  weekly_active_days: "number",
  avg_session_time: "number",
  features_used: "number",
  usage_growth_rate: "number",
  last_login_days_ago: "number",
  monthly_fee: "number",
  total_revenue: "number",
  payment_method: "text",
  payment_failures: "number",
  discount_applied: "select",
  price_increase_last_3m: "select",
  support_tickets: "number",
  avg_resolution_time: "number",
  complaint_type: "text",
  csat_score: "number",
  escalations: "number",
  email_open_rate: "number",
  marketing_click_rate: "number",
  nps_score: "number",
  survey_response: "text",
  referral_count: "number",
};

const selectOptions: Record<string, string[]> = {
  gender: ["Male", "Female", "Other"],
  contract_type: ["Monthly", "Yearly", "Two-Year"],
  discount_applied: ["Yes", "No"],
  price_increase_last_3m: ["Yes", "No"],
};

export default function ChurnPredictorForm() {
  const [currentStep, setCurrentStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [formData, setFormData] = useState<FormData>({
    gender: "",
    age: "",
    customer_segment: "",
    tenure_months: "",
    signup_channel: "",
    contract_type: "",
    monthly_logins: "",
    weekly_active_days: "",
    avg_session_time: "",
    features_used: "",
    usage_growth_rate: "",
    last_login_days_ago: "",
    monthly_fee: "",
    total_revenue: "",
    payment_method: "",
    payment_failures: "",
    discount_applied: "",
    price_increase_last_3m: "",
    support_tickets: "",
    avg_resolution_time: "",
    complaint_type: "",
    csat_score: "",
    escalations: "",
    email_open_rate: "",
    marketing_click_rate: "",
    nps_score: "",
    survey_response: "",
    referral_count: "",
  });

  const handleInputChange = (field: keyof FormData, value: string | number) => {
    setFormData((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    }
  };

  const handlePrev = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const payload = {
        gender: formData.gender,
        age: Number(formData.age),
        customer_segment: formData.customer_segment,
        tenure_months: Number(formData.tenure_months),
        signup_channel: formData.signup_channel,
        contract_type: formData.contract_type,
        monthly_logins: Number(formData.monthly_logins),
        weekly_active_days: Number(formData.weekly_active_days),
        avg_session_time: Number(formData.avg_session_time),
        features_used: Number(formData.features_used),
        usage_growth_rate: Number(formData.usage_growth_rate) / 100,
        last_login_days_ago: Number(formData.last_login_days_ago),
        monthly_fee: Number(formData.monthly_fee),
        total_revenue: Number(formData.total_revenue),
        payment_method: formData.payment_method,
        payment_failures: Number(formData.payment_failures),
        discount_applied: formData.discount_applied,
        price_increase_last_3m: formData.price_increase_last_3m,
        support_tickets: Number(formData.support_tickets),
        avg_resolution_time: Number(formData.avg_resolution_time),
        complaint_type: formData.complaint_type || null,
        csat_score: formData.csat_score ? Number(formData.csat_score) : null,
        escalations: Number(formData.escalations),
        email_open_rate: Number(formData.email_open_rate) / 100,
        marketing_click_rate: Number(formData.marketing_click_rate) / 100,
        nps_score: Number(formData.nps_score),
        survey_response: formData.survey_response || null,
        referral_count: Number(formData.referral_count),
      };

      const response = await axios.post(
        "http://localhost:8000/predict",
        payload,
      );
      setResult(response.data);
    } catch (err) {
      setError(
        axios.isAxiosError(err)
          ? err.response?.data?.message ||
              err.message ||
              "Erreur lors de la prédiction"
          : err instanceof Error
            ? err.message
            : "Une erreur s'est produite. Veuillez vérifier l'API.",
      );
    } finally {
      setLoading(false);
    }
  };

  const currentStepData = steps[currentStep];

  if (result) {
    return (
      <div className="min-h-screen bg-emerald-600 flex items-center justify-center p-4 selection:bg-emerald-200">
        <Card className="w-full max-w-2xl bg-white shadow-2xl rounded-2xl overflow-hidden border-0">
          <div className="p-10">
            <div className="text-center mb-8">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-emerald-100 mb-4">
                <span className="text-3xl">✨</span>
              </div>
              <h2 className="text-3xl font-bold text-gray-900 mb-2">
                Analyse terminée
              </h2>
              <p className="text-gray-500">
                Voici l'évaluation du risque de désabonnement pour ce client
              </p>
            </div>

            <div className="grid grid-cols-2 gap-5">
              <div className="bg-gray-50 p-6 rounded-xl border border-gray-100">
                <p className="text-sm text-gray-500 font-semibold uppercase tracking-wider mb-1">
                  Prédiction
                </p>
                <div className="flex items-center gap-2">
                  <span className="text-2xl">
                    {result.churn === 0 ? "✅" : "⚠️"}
                  </span>
                  <p className="text-2xl font-bold text-gray-900">
                    {result.churn === 0 ? "Fidèle" : "Départ"}
                  </p>
                </div>
              </div>
              <div className="bg-gray-50 p-6 rounded-xl border border-gray-100">
                <p className="text-sm text-gray-500 font-semibold uppercase tracking-wider mb-1">
                  Probabilité
                </p>
                <p className="text-3xl font-bold text-gray-900">
                  {(result.probability * 100).toFixed(1)}
                  <span className="text-xl text-gray-400">%</span>
                </p>
              </div>

              <div className="bg-gray-50 p-6 rounded-xl border border-gray-100">
                <p className="text-sm text-gray-500 font-semibold uppercase tracking-wider mb-1">
                  Seuil
                </p>
                <p className="text-3xl font-bold text-gray-900">
                  {(result.threshold * 100).toFixed(1)}
                  <span className="text-xl text-gray-400">%</span>
                </p>
              </div>
              <div
                className={`p-6 rounded-xl border ${
                  result.risk_level === "low"
                    ? "bg-emerald-50 border-emerald-200"
                    : result.risk_level === "medium"
                      ? "bg-amber-50 border-amber-200"
                      : "bg-rose-50 border-rose-200"
                }`}
              >
                <p
                  className={`text-sm font-semibold uppercase tracking-wider mb-1 ${
                    result.risk_level === "low"
                      ? "text-emerald-700"
                      : result.risk_level === "medium"
                        ? "text-amber-700"
                        : "text-rose-700"
                  }`}
                >
                  Risque
                </p>
                <p
                  className={`text-3xl font-bold ${
                    result.risk_level === "low"
                      ? "text-emerald-900"
                      : result.risk_level === "medium"
                        ? "text-amber-900"
                        : "text-rose-900"
                  }`}
                >
                  {result.risk_level === "low"
                    ? "Faible"
                    : result.risk_level === "medium"
                      ? "Moyen"
                      : "Élevé"}
                </p>
              </div>
            </div>

            <Button
              onClick={() => {
                setResult(null);
                setCurrentStep(0);
                setFormData({
                  gender: "",
                  age: "",
                  customer_segment: "",
                  tenure_months: "",
                  signup_channel: "",
                  contract_type: "",
                  monthly_logins: "",
                  weekly_active_days: "",
                  avg_session_time: "",
                  features_used: "",
                  usage_growth_rate: "",
                  last_login_days_ago: "",
                  monthly_fee: "",
                  total_revenue: "",
                  payment_method: "",
                  payment_failures: "",
                  discount_applied: "",
                  price_increase_last_3m: "",
                  support_tickets: "",
                  avg_resolution_time: "",
                  complaint_type: "",
                  csat_score: "",
                  escalations: "",
                  email_open_rate: "",
                  marketing_click_rate: "",
                  nps_score: "",
                  survey_response: "",
                  referral_count: "",
                });
              }}
              className="w-full mt-8 bg-gray-900 hover:bg-gray-800 text-white rounded-xl h-14 text-base font-medium transition-all"
            >
              Nouvelle analyse
            </Button>
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-emerald-600 flex items-center justify-center p-4 py-12 selection:bg-emerald-200">
      <Card className="w-full max-w-3xl bg-white shadow-2xl rounded-2xl overflow-hidden border-0">
        <div className="p-8 sm:p-12">
          <div className="mb-10 text-center">
            <h1 className="text-3xl sm:text-4xl font-extrabold text-gray-900 tracking-tight mb-3">
              Analyse Client
            </h1>
            <p className="text-gray-500 text-lg">
              Estimez la probabilité de désabonnement de votre client
            </p>
          </div>

          <div className="mb-10">
            <div className="flex justify-between items-center mb-4 relative">
              <div className="absolute left-0 top-1/2 -translate-y-1/2 w-full h-1 bg-gray-100 rounded-full z-0"></div>
              <div
                className="absolute left-0 top-1/2 -translate-y-1/2 h-1 bg-emerald-500 transition-all duration-500 z-0 rounded-full"
                style={{
                  width: `${(currentStep / (steps.length - 1)) * 100}%`,
                }}
              ></div>

              {steps.map((step, index) => {
                const isActive = index === currentStep;
                const isCompleted = index < currentStep;
                return (
                  <div
                    key={index}
                    className="relative z-10 flex flex-col items-center"
                  >
                    <div
                      className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold transition-all duration-300 ${
                        isActive
                          ? "bg-emerald-500 text-white shadow-md ring-4 ring-emerald-50"
                          : isCompleted
                            ? "bg-emerald-500 text-white"
                            : "bg-white text-gray-300 border-2 border-gray-200"
                      }`}
                    >
                      {isCompleted ? "✓" : index + 1}
                    </div>
                  </div>
                );
              })}
            </div>
            <div className="text-center mt-6">
              <span className="text-sm font-bold text-emerald-600 uppercase tracking-widest">
                Étape {currentStep + 1} / {steps.length}
              </span>
              <h3 className="text-xl font-semibold text-gray-900 mt-1">
                {currentStepData.title}
              </h3>
            </div>
          </div>

          <div className="space-y-5 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
              {currentStepData.fields.map((field) => {
                const fieldName = field as keyof FormData;
                const fieldType = fieldTypes[field];
                const isFullWidth = [
                  "customer_segment",
                  "complaint_type",
                  "survey_response",
                ].includes(field);

                return (
                  <div
                    key={field}
                    className={isFullWidth ? "sm:col-span-2" : ""}
                  >
                    <Label
                      htmlFor={field}
                      className="text-sm font-semibold text-gray-700 block mb-2"
                    >
                      {fieldLabels[field]}
                    </Label>

                    {fieldType === "select" ? (
                      <Select
                        value={String(formData[fieldName] || "")}
                        onValueChange={(value) =>
                          handleInputChange(fieldName, value)
                        }
                      >
                        <SelectTrigger
                          id={field}
                          className="w-full bg-gray-50 border-gray-200 focus:bg-white focus:border-emerald-500 focus:ring-emerald-500/20 rounded-xl h-12 text-gray-900 transition-colors"
                        >
                          <SelectValue placeholder="Sélectionner..." />
                        </SelectTrigger>
                        <SelectContent className="bg-white rounded-xl border-gray-100 shadow-xl">
                          {selectOptions[field]?.map((option) => (
                            <SelectItem
                              key={option}
                              value={option}
                              className="focus:bg-emerald-50 focus:text-emerald-900 cursor-pointer"
                            >
                              {option}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    ) : fieldType === "number" ? (
                      <Input
                        id={field}
                        type="number"
                        step="0.01"
                        value={formData[fieldName]}
                        onChange={(e) =>
                          handleInputChange(
                            fieldName,
                            e.target.value === "" ? "" : Number(e.target.value),
                          )
                        }
                        placeholder="0"
                        className="w-full bg-gray-50 border-gray-200 focus:bg-white focus:border-emerald-500 focus:ring-2 focus:ring-emerald-500/20 rounded-xl h-12 text-gray-900 transition-colors"
                      />
                    ) : (
                      <Input
                        id={field}
                        type="text"
                        value={formData[fieldName]}
                        onChange={(e) =>
                          handleInputChange(fieldName, e.target.value)
                        }
                        placeholder="..."
                        className="w-full bg-gray-50 border-gray-200 focus:bg-white focus:border-emerald-500 focus:ring-2 focus:ring-emerald-500/20 rounded-xl h-12 text-gray-900 transition-colors"
                      />
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {error && (
            <div className="mt-6 p-4 rounded-xl bg-red-50 border border-red-100 text-red-600 text-sm font-medium flex items-start gap-3">
              <span className="text-red-500">⚠️</span>
              {error}
            </div>
          )}

          <div className="flex gap-4 mt-10 pt-6 border-t border-gray-100">
            <Button
              onClick={handlePrev}
              disabled={currentStep === 0}
              variant="outline"
              className={`flex-1 rounded-xl h-14 font-semibold text-base transition-all ${
                currentStep === 0
                  ? "opacity-0 cursor-default"
                  : "bg-white border-gray-200 text-gray-600 hover:bg-gray-50 hover:text-gray-900"
              }`}
            >
              Retour
            </Button>

            {currentStep < steps.length - 1 ? (
              <Button
                onClick={handleNext}
                className="flex-1 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl h-14 font-semibold text-base shadow-sm shadow-emerald-600/20 transition-all"
              >
                Continuer
              </Button>
            ) : (
              <Button
                onClick={handleSubmit}
                disabled={loading}
                className="flex-1 bg-gray-900 hover:bg-gray-800 text-white rounded-xl h-14 font-semibold text-base shadow-sm transition-all relative overflow-hidden"
              >
                {loading ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg
                      className="animate-spin h-5 w-5 text-white"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      ></circle>
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      ></path>
                    </svg>
                    Analyse...
                  </span>
                ) : (
                  "Lancer l'analyse ✨"
                )}
              </Button>
            )}
          </div>
        </div>
      </Card>
    </div>
  );
}
