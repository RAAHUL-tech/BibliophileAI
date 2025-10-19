import { useState } from "react";
import PreferencesGenre from "./PreferencesGenre";
import PreferencesAuthors from "./PreferencesAuthors";
import DemographicForm from "./DemographicForm";

type UserOnboardingProps = {
  token: string;
  onComplete: () => void;
};

export default function UserOnboarding({ token, onComplete }: UserOnboardingProps) {
  // Step 0: genres, 1: authors, 2: demographics
  const [step, setStep] = useState(0);
  const [genres, setGenres] = useState<string[]>([]);
  const [authors, setAuthors] = useState<string[]>([]);
  const [pincode, setPincode] = useState<string>("");
  const [age, setAge] = useState<number | undefined>();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const next = () => setStep((s) => s + 1);
  const prev = () => setStep((s) => s - 1);

  const handleGenres = (selected: string[]) => {
    setGenres(selected);
    next();
  };

  const handleAuthors = (selected: string[]) => {
    setAuthors(selected);
    next();
  };

  const handleDemographics = async (values: { pincode: string; age: number }) => {
  setLoading(true);
  setError(null);
  try {
    const res = await fetch("http://localhost:8000/api/v1/user/preferences", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({
        genres,
        authors,
        pincode: values.pincode,
        age: values.age,
      }),
    });
    if (!res.ok) {
      const err = await res.json();
      setError(err.detail || "Failed to save preferences");
    } else {
      onComplete();
    }
  } catch {
    setError("Network error");
  } finally {
    setLoading(false);
  }
};

  return (
    <div>
      {step === 0 && (
        <PreferencesGenre initialSelectedGenres={genres} onSave={handleGenres} loading={loading} />
      )}
      {step === 1 && (
        <PreferencesAuthors
          initialSelectedAuthors={authors}
          onSave={handleAuthors}
          onBack={prev}
          loading={loading}
          token={token}
        />
      )}
      {step === 2 && (
        <DemographicForm
          initialAge={age}
          initialPincode={pincode}
          onSave={handleDemographics}
          onBack={prev}
          loading={loading}
        />
      )}
      {error && <div className="alert alert-danger mt-2">{error}</div>}
    </div>
  );
}
