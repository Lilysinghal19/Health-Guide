import tkinter as tk
from tkinter import messagebox, ttk

from health_logic import HealthInput, analyze_health_info, build_response


SYMPTOMS = [
    "Fever",
    "Cough",
    "Headache",
    "Fatigue",
    "Sore throat",
    "Runny nose",
    "Body aches",
    "Nausea",
    "Vomiting",
    "Diarrhea",
    "Chest pain",
    "Dizziness",
]

CONDITIONS = [
    "None",
    "Diabetes",
    "Hypertension",
    "Asthma",
    "Heart disease",
    "Pregnancy",
    "Kidney disease",
    "Immune system condition",
]

DURATIONS = [
    "Less than 1 day",
    "1 to 3 days",
    "4 to 7 days",
    "More than 1 week",
    "More than 2 weeks",
]

SEVERITIES = ["Mild", "Moderate", "Severe"]


class HealthGuideUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("HealthGuide Chatbot")
        self.root.geometry("980x720")
        self.root.minsize(820, 620)

        self.symptom_vars: dict[str, tk.BooleanVar] = {}
        self.condition_vars: dict[str, tk.BooleanVar] = {}
        self.age_var = tk.StringVar()
        self.duration_var = tk.StringVar(value=DURATIONS[1])
        self.severity_var = tk.StringVar(value=SEVERITIES[0])
        self.breathing_var = tk.StringVar(value="No")

        self._build_layout()

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(1, weight=1)

        title = ttk.Label(
            self.root,
            text="HealthGuide Chatbot",
            font=("Segoe UI", 20, "bold"),
        )
        title.grid(row=0, column=0, columnspan=2, padx=18, pady=(18, 4), sticky="w")

        note = ttk.Label(
            self.root,
            text=(
                "Enter symptoms and health details. The chatbot gives non-medication "
                "self-care guidance and flags warning signs."
            ),
            wraplength=900,
        )
        note.grid(row=1, column=0, columnspan=2, padx=18, pady=(0, 12), sticky="nw")

        form_frame = ttk.Frame(self.root, padding=18)
        form_frame.grid(row=2, column=0, padx=(18, 9), pady=(0, 18), sticky="nsew")

        output_frame = ttk.Frame(self.root, padding=18)
        output_frame.grid(row=2, column=1, padx=(9, 18), pady=(0, 18), sticky="nsew")
        self.root.rowconfigure(2, weight=1)

        self._build_form(form_frame)
        self._build_output(output_frame)

    def _build_form(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)

        ttk.Label(parent, text="Symptoms", font=("Segoe UI", 11, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        self._add_checkbox_grid(parent, SYMPTOMS, self.symptom_vars, start_row=1)

        condition_row = 7
        ttk.Label(parent, text="Existing conditions", font=("Segoe UI", 11, "bold")).grid(
            row=condition_row, column=0, columnspan=2, pady=(14, 0), sticky="w"
        )
        self._add_checkbox_grid(
            parent,
            CONDITIONS,
            self.condition_vars,
            start_row=condition_row + 1,
            command=self._handle_condition_change,
        )

        details_row = condition_row + 6
        ttk.Label(parent, text="Age (optional)").grid(
            row=details_row, column=0, pady=(14, 0), sticky="w"
        )
        ttk.Entry(parent, textvariable=self.age_var).grid(
            row=details_row + 1, column=0, padx=(0, 8), sticky="ew"
        )

        ttk.Label(parent, text="Breathing difficulty").grid(
            row=details_row, column=1, pady=(14, 0), sticky="w"
        )
        ttk.Combobox(
            parent,
            textvariable=self.breathing_var,
            values=["No", "Yes"],
            state="readonly",
        ).grid(row=details_row + 1, column=1, sticky="ew")

        ttk.Label(parent, text="Duration of symptoms").grid(
            row=details_row + 2, column=0, pady=(12, 0), sticky="w"
        )
        ttk.Combobox(
            parent,
            textvariable=self.duration_var,
            values=DURATIONS,
            state="readonly",
        ).grid(row=details_row + 3, column=0, padx=(0, 8), sticky="ew")

        ttk.Label(parent, text="Severity").grid(
            row=details_row + 2, column=1, pady=(12, 0), sticky="w"
        )
        ttk.Combobox(
            parent,
            textvariable=self.severity_var,
            values=SEVERITIES,
            state="readonly",
        ).grid(row=details_row + 3, column=1, sticky="ew")

        ttk.Label(parent, text="Other details").grid(
            row=details_row + 4, column=0, columnspan=2, pady=(12, 0), sticky="w"
        )
        self.details_text = tk.Text(parent, height=4, wrap="word")
        self.details_text.grid(row=details_row + 5, column=0, columnspan=2, sticky="ew")

        actions = ttk.Frame(parent)
        actions.grid(row=details_row + 6, column=0, columnspan=2, pady=(16, 0), sticky="ew")
        ttk.Button(actions, text="Analyze Symptoms", command=self._analyze).pack(
            side="left"
        )
        ttk.Button(actions, text="Clear", command=self._clear).pack(side="left", padx=8)

    def _build_output(self, parent: ttk.Frame) -> None:
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)

        ttk.Label(parent, text="Bot Response", font=("Segoe UI", 11, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        self.response_text = tk.Text(parent, wrap="word", state="disabled")
        self.response_text.grid(row=1, column=0, pady=(8, 0), sticky="nsew")
        self._set_response(
            "Complete the form to receive non-medication self-care guidance and safety advice."
        )

    def _add_checkbox_grid(
        self,
        parent: ttk.Frame,
        labels: list[str],
        store: dict[str, tk.BooleanVar],
        start_row: int,
        command=None,
    ) -> None:
        for index, label in enumerate(labels):
            var = tk.BooleanVar(value=label == "None")
            store[label] = var
            checkbox = ttk.Checkbutton(parent, text=label, variable=var, command=command)
            checkbox.grid(
                row=start_row + index // 2,
                column=index % 2,
                padx=(0, 8),
                pady=2,
                sticky="w",
            )

    def _handle_condition_change(self) -> None:
        none_selected = self.condition_vars["None"].get()
        selected_conditions = [
            label for label, var in self.condition_vars.items() if var.get() and label != "None"
        ]

        if none_selected and selected_conditions:
            self.condition_vars["None"].set(False)
        elif not selected_conditions:
            self.condition_vars["None"].set(True)

    def _analyze(self) -> None:
        symptoms = [label for label, var in self.symptom_vars.items() if var.get()]
        conditions = [label for label, var in self.condition_vars.items() if var.get()]
        age = self._parse_age()

        if not symptoms:
            messagebox.showwarning("Missing symptoms", "Please select at least one symptom.")
            return
        if age == "invalid":
            messagebox.showwarning("Invalid age", "Please enter a valid age or leave it blank.")
            return

        info = HealthInput(
            symptoms=symptoms,
            conditions=conditions or ["None"],
            age=age,
            duration=self.duration_var.get(),
            severity=self.severity_var.get(),
            breathing_difficulty=self.breathing_var.get() == "Yes",
            details=self.details_text.get("1.0", "end").strip(),
        )
        analysis = analyze_health_info(info)
        self._set_response(build_response(analysis))

    def _parse_age(self) -> int | None | str:
        raw_age = self.age_var.get().strip()
        if not raw_age:
            return None
        if not raw_age.isdigit():
            return "invalid"
        age = int(raw_age)
        if age < 0 or age > 120:
            return "invalid"
        return age

    def _clear(self) -> None:
        for var in self.symptom_vars.values():
            var.set(False)
        for label, var in self.condition_vars.items():
            var.set(label == "None")
        self.age_var.set("")
        self.duration_var.set(DURATIONS[1])
        self.severity_var.set(SEVERITIES[0])
        self.breathing_var.set("No")
        self.details_text.delete("1.0", "end")
        self._set_response(
            "Complete the form to receive non-medication self-care guidance and safety advice."
        )

    def _set_response(self, text: str) -> None:
        self.response_text.configure(state="normal")
        self.response_text.delete("1.0", "end")
        self.response_text.insert("1.0", text)
        self.response_text.configure(state="disabled")


def run_app() -> None:
    root = tk.Tk()
    HealthGuideUI(root)
    root.mainloop()
