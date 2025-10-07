import { useState } from "react";

type DemographicProps = {
  initialPincode?: string;
  initialAge?: number;
  onSave: (values: { pincode: string; age: number }) => void;
  onBack: () => void;
  loading?: boolean;
};

export default function DemographicForm({
  initialPincode = "",
  initialAge,
  onSave,
  onBack,
  loading = false,
}: DemographicProps) {
  const [pincode, setPincode] = useState(initialPincode);
  const [age, setAge] = useState<number | "">(initialAge ?? "");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!pincode || !age) return;
    onSave({ pincode, age: Number(age) });
  };

  return (
    <form className="p-4" onSubmit={handleSubmit}>
      <h3>Tell us about yourself</h3>
      <div className="mb-3">
        <label className="form-label">Pincode</label>
        <input className="form-control"
          value={pincode} onChange={e => setPincode(e.target.value)} required minLength={5} maxLength={10}
          disabled={loading} />
      </div>
      <div className="mb-4">
        <label className="form-label">Age</label>
        <input className="form-control"
          type="number" min={10} max={120}
          value={age} onChange={e => setAge(Number(e.target.value))}
          required
          disabled={loading} />
      </div>
      <div>
        <button className="btn btn-secondary me-2" type="button" onClick={onBack} disabled={loading}>Back</button>
        <button className="btn btn-success" type="submit" disabled={!pincode || !age || loading}>
          {loading ? "Saving..." : "Finish"}
        </button>
      </div>
    </form>
  );
}
