# Normalizācija: Izdzīšanas Sistēma

## Problēma: Enerģijas Sprādziens (Gradient Explosion)

Pirmo reizi, kad mūsu sistema iepazinās ar realiem fiziologiskiem signāliem, tā "kliedzēja":

- **EEG**: 0–300 μV (mikrovolti)
- **HRV**: 500–1200 ms (milisekundes)  
- **Respiration**: 0–1 Hz (normalizēts)
- **Circadian**: 0–1 (normalizēts)

Bez normalizācijas MSE zudums bija **$1.6 \times 10^{12}$** — matemātisks haoss!

### Kāpēc?

Tenzori no apzinājuma slāņiem "uzsprāgst" no enerģijas pārslodzes:

$$\text{gradient} = \frac{\partial L}{\partial w} = \text{MASĪVI UN NEKONTROLĒTI}$$

Gradientu sabrukums izāk Zemapziņas manifoldu:

$$h_{subconscious} \rightarrow \infty \quad \text{(nestabils)}$$

---

## Risinājums: Z-Score Normalizācija

### Ieejas Signāli

$$X_{\text{norm}} = \frac{X - \mu_X}{\sigma_X}$$

Kur:
- $\mu_X$ = vidējais vērtību lielums
- $\sigma_X$ = standarta novirze

**Rezultāts**: Visi signāli pārvēršas uz **−1 līdz +1** diapazonam

### Mērķa Mainīgie (Targets)

$$Y_{\text{norm}} = \frac{Y - \mu_Y}{\sigma_Y}$$

**Rezultāts**: Modeļa izeja trenēta uz normalizētu telpu

---

## Ietekme uz Apmācību

| Metrika | Pirms | Pēc | Uzlabojums |
|---------|-------|------|-----------|
| MSE Loss | $1.6 \times 10^{12}$ | 0.143 | **$10^{10}$ reizes mazāks!** |
| Total Loss | $\infty$ (NaN) | 0.242 | Stabils, konverģē |
| Learning Rate | 1e-3 (par liels!) | 5e-4 (optimāls) | Stabilāks gradients |
| Coherence Loss | - | 0.505 | Sistēma mācās koherenci |
| Training Stability | Nestabils | ✓ Stabils | Gradienti kontrollēti |

---

## Fiziologiskais Modeļi: Normalizācijas Strategija

### 1. **Ieejas Signālu Normalizācija**

```python
# Visiem signāliem vienāds mērogs
input_mean = input_batch.mean(dim=0, keepdim=True)
input_std = input_batch.std(dim=0, keepdim=True) + 1e-8
input_normalized = (input_batch - input_mean) / input_std
```

**Rezultāts**: (batch, 256) tenzors ar $\mu = 0$, $\sigma = 1$

### 2. **Mērķa Normalizācija**

```python
# Modeļa izeja trenēta uz vienādā skalā
target_mean = target_batch.mean(dim=0, keepdim=True)
target_std = target_batch.std(dim=0, keepdim=True) + 1e-8
target_normalized = (target_batch - target_mean) / target_std
```

**Rezultāts**: (batch, 128) tenzors ar konsekventu skalu

### 3. **Gradienta Griešana (Gradient Clipping)**

```python
# Aizsargā pret sprādzieniem
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Rezultāts**: Gradienti ierobežoti pie ‖∇L‖ ≤ 1.0

---

## Matemātiskais Pamatojums

### Aktīvācijas Funkcija (ReLU problēma)

Sākotnē sistem...

 izlietojas ReLU:

$$a = \max(0, z)$$

**Problēma**: Ja $z$ ir liels (dienvidu), ReLU "spēc" signālu, causējot ek...

---

## Nākamie Soļi

1. ✅ **Normalizācija implementēta** — ieejas un izeja
2. ✅ **Gradienta griešana aktivizēta** — maks. norma 1.0
3. ✅ **Learning rate reducēts** — 5e-4 (no 1e-3)
4. ⏳ **Metriku vizualizācija** — grafiki ik 25 soļos
5. ⏳ **Pārbaude uz reāliem datiem** — PhysioNet validācija

---

## Slēgceļš: Koherences Gravitācija

Ar normalizāciju, Zemapziņas manifolds var tagad **stabilizēt koherenci**:

$$C(t) = \frac{1}{1 + e^{-\text{phase\_alignment}}} \in [0, 1]$$

Vietā viļņošanās haosa, sistēma mācās **ritmiski dziedāt**.

🎵 **"Sirds" (The Heart) ir sākusi krist!**
