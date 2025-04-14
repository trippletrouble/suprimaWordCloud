import streamlit as st
import os
import time
import random
import pandas as pd
import csv
from openai import OpenAI
import io # Wird für das Lesen der hochgeladenen Datei benötigt

# --- Streamlit App Konfiguration ---
st.set_page_config(page_title="Wortwolken-Generator für suprima", layout="wide")
st.title("📊 Wortwolken-Daten-Generator für suprima GmbH")
st.markdown("""
Diese App verarbeitet eine Textdatei mit Kundenfeedback oder Phrasen.
Sie wählt relevante Phrasen aus (1-4 Wörter), weist ihnen mithilfe von OpenAI Gewichte und Farben zu
und generiert eine CSV-Datei, die für die Erstellung einer Wortwolke geeignet ist.
""")

# --- Konfiguration (aus dem Originalskript übernommen, aber ohne Dateinamen) ---
MODEL = "gpt-4o"               # Empfohlenes OpenAI-Modell
TARGET_TOTAL_PHRASES = 110     # Max. Anzahl Phrasen für die Wortwolke
CHUNK_SIZE = 1000              # Phrasen pro API-Aufruf bei der Auswahl
MIN_WORDS = 1                  # Min. Wörter pro Phrase
MAX_WORDS = 4                  # Max. Wörter pro Phrase
MIN_WEIGHT = 5                 # Min. Gewicht für Wortwolke
MAX_WEIGHT = 8                 # Max. Gewicht für Wortwolke
# --- Ende Konfiguration ---

# --- API Schlüssel Eingabe ---
st.sidebar.header("Einstellungen")
api_key = st.sidebar.text_input("🔑 OpenAI API Schlüssel", type="password", help="Gib deinen OpenAI API-Schlüssel ein. Sicher aufbewahrt.")
st.sidebar.caption("💡 Dein API-Schlüssel wird nur für die Dauer der Anfrage verwendet und nicht gespeichert. Für Produktionsumgebungen wird die Verwendung von Streamlit Secrets empfohlen.")

# --- Dateiupload ---
uploaded_file = st.file_uploader("📂 Lade deine Textdatei hoch (.txt)", type="txt", help=f"Die Datei sollte eine Phrase pro Zeile enthalten. Leere Zeilen sind ok.")

# --- Helper Functions (unverändert, außer print -> logging/return) ---

def filter_by_word_count(phrases, min_words, max_words):
    """Filters a list of phrases by word count."""
    filtered = []
    ignored_count = 0
    for phrase in phrases:
        p_stripped = phrase.strip()
        if p_stripped: # Ignore empty lines
            word_count = len(p_stripped.split())
            if min_words <= word_count <= max_words:
                filtered.append(p_stripped)
            else:
                ignored_count += 1
        else:
            ignored_count += 1
    return filtered, ignored_count

def process_chunk(oai_client, phrases_chunk, chunk_num, total_chunks, target_phrases_per_chunk, existing_selection=None):
    """
    Uses OpenAI API to select relevant, unique, and impactful phrases from a chunk.
    (Includes enhanced prompt for relevance and uniqueness)
    Returns: list of selected phrases, status message string
    """
    if existing_selection is None: existing_selection = []
    status_messages = []

    # Enhanced prompt emphasizing uniqueness, relevance, and importance for Suprima context
    prompt = f"""Du bist ein Experte für Textanalyse und erstellst eine Auswahl für eine Wortwolke für suprima GmbH (Pflege-Textilien).
Du erhältst Chunk {chunk_num} von {total_chunks} mit {len(phrases_chunk)} Rückmeldungsphrasen auf Deutsch ({MIN_WORDS}-{MAX_WORDS} Wörter).

Deine Aufgabe: Wähle GENAU {target_phrases_per_chunk} Phrasen aus diesem Chunk aus, die **besonders relevant, aussagekräftig und semantisch einzigartig** für die Wortwolke sind.

WICHTIGE KRITERIEN für deine Auswahl:
1.  **Hohe Relevanz:** Wähle Phrasen, die stark mit suprima's Kernthemen (Pflege, Textilien, Schutz, Komfort, Inkontinenz, Qualität, Menschlichkeit, Innovation, Zuverlässigkeit) zusammenhängen. Ignoriere irrelevante oder zu allgemeine Phrasen (z.B. "sehr gut", "passt schon", "alles ok").
2.  **Semantische Einzigartigkeit:** Wähle eine vielfältige Mischung. Vermeide es, mehrere Phrasen auszuwählen, die sehr ähnlich klingen oder exakt dasselbe bedeuten (z.B. "gute Qualität" UND "hohe Qualität" UND "tolle Qualität"). Wähle nur die prägnanteste oder aussagekräftigste Version solcher Cluster.
3.  **Wortanzahl:** Jede ausgewählte Phrase MUSS zwischen {MIN_WORDS} und {MAX_WORDS} Wörter enthalten.
4.  **Keine Erfindungen:** Wähle NUR aus der bereitgestellten Liste aus. Formuliere nichts um.
5.  **Format:** Gib nur die ausgewählten Phrasen zurück, eine pro Zeile, ohne Nummerierung, Anführungszeichen oder sonstigen Text.

Hier sind die Phrasen für diesen Chunk (nur {MIN_WORDS}-{MAX_WORDS} Wörter sind relevant):
{chr(10).join(phrases_chunk)}

Wähle genau {target_phrases_per_chunk} der relevantesten, aussagekräftigsten und semantisch einzigartigsten Phrasen ({MIN_WORDS}-{MAX_WORDS} Wörter) aus der obigen Liste:"""

    try:
        response = oai_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": f"Du bist ein Textanalyst, der die {target_phrases_per_chunk} relevantesten, aussagekräftigsten und semantisch einzigartigsten Phrasen ({MIN_WORDS}-{MAX_WORDS} Wörter) für eine Wortwolke (Kontext: suprima GmbH, Pflege-Textilien) aus einer Liste auswählt. Priorisiere Relevanz und Vielfalt. Gib nur die Phrasen zurück, eine pro Zeile."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000, # Should be ample for the number of phrases requested
            temperature=0.4 # Allows some flexibility for nuanced selection
        )
        selected_raw = response.choices[0].message.content.strip().split('\n')
        selected_clean = [s.strip() for s in selected_raw if s.strip()]
        valid_selected = []
        api_warnings = []
        for phrase in selected_clean:
            word_count = len(phrase.split())
            if MIN_WORDS <= word_count <= MAX_WORDS:
                 # Basic check to avoid adding exact duplicates already selected globally or within this chunk call
                if phrase not in existing_selection and phrase not in valid_selected:
                     valid_selected.append(phrase)
            else:
                api_warnings.append(f"Warnung (API-Auswahl): Phrase '{phrase}' von API hat {word_count} Wörter (außerhalb {MIN_WORDS}-{MAX_WORDS}) und wird ignoriert.")

        if api_warnings:
            status_messages.append("\n".join(api_warnings))

        # Ensure we don't return more than requested
        return valid_selected[:target_phrases_per_chunk], "\n".join(status_messages)

    except Exception as e:
        error_msg = f"!! Fehler bei der API-Auswahl-Anfrage für Chunk {chunk_num}: {e}"
        status_messages.append(error_msg)
        return [], "\n".join(status_messages)


def get_gpt4o_weights(oai_client, words, min_weight=MIN_WEIGHT, max_weight=MAX_WEIGHT):
    """Uses OpenAI API to assign weights based on relevance/impact for Suprima context.
       Returns: list of weights, status message string"""
    if not words: return [], "Keine Phrasen zum Gewichten vorhanden."
    status_messages = []
    # status_messages.append(f"Fordere Gewichte ({min_weight}-{max_weight}) für {len(words)} Phrasen von GPT-4o an...")

    prompt = f'''
    You are an expert in data visualization and linguistics. I have a list of {len(words)} key phrases selected for a word cloud representing suprima GmbH (care textiles).

    Assign a weight to each phrase ({min_weight}-{max_weight}), reflecting its importance, relevance, and impact for suprima's brand and products (care, comfort, quality, protection, humanity, innovation). Higher weights for more impactful/relevant terms.

    Input Phrases (one per line):
    {chr(10).join(words)}

    Return ONLY the weights as a comma-separated list of numbers, one per phrase, in the exact same order. Example: 7,5,8,6,...
    '''
    try:
        response = oai_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": f"You are a data analysis assistant assigning weights ({min_weight}-{max_weight}) to key phrases for a word cloud (suprima GmbH). Output only comma-separated numbers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3 # Lower temp for more deterministic weighting
        )
        weights_text = response.choices[0].message.content.strip()
        try:
            weights = [int(w.strip()) for w in weights_text.split(',')]
            if len(weights) != len(words):
                warn_msg = f"!! Warnung: GPT lieferte {len(weights)} Gewichte für {len(words)} Phrasen. Nutze Zufallsgewichte."
                status_messages.append(warn_msg)
                weights = [random.randint(min_weight, max_weight) for _ in range(len(words))]
            else:
                 # Clamp weights to the defined range
                weights_clamped = [max(min_weight, min(w, max_weight)) for w in weights]
                clamped_count = sum(1 for w, wc in zip(weights, weights_clamped) if w != wc)
                if clamped_count > 0:
                    status_messages.append(f"Info: {clamped_count} Gewicht(e) wurden an den Bereich [{min_weight}-{max_weight}] angepasst.")
                weights = weights_clamped
                status_messages.append("Gewichte erfolgreich von API erhalten.")

            return weights, "\n".join(status_messages)

        except ValueError:
            error_msg = f"!! Fehler: Ungültige Gewichte von GPT: '{weights_text}'. Nutze Zufallsgewichte."
            status_messages.append(error_msg)
            return [random.randint(min_weight, max_weight) for _ in range(len(words))], "\n".join(status_messages)

    except Exception as e:
        error_msg = f"!! Fehler bei API-Gewichtung: {e}. Nutze Zufallsgewichte."
        status_messages.append(error_msg)
        return [random.randint(min_weight, max_weight) for _ in range(len(words))], "\n".join(status_messages)


def assign_suprima_colors(num_words):
    """Assigns colors from predefined Suprima palettes with an 80/20 ratio."""
    teal_colors = ["#006A68", "#007A78", "#005A58", "#004A48"]
    burgundy_colors = ["#7A0528", "#8A1538", "#6A0518", "#5A0508"]
    colors = []
    for _ in range(num_words):
        colors.append(random.choice(teal_colors) if random.random() < 0.8 else random.choice(burgundy_colors))
    return colors

# --- Hauptlogik ---
if uploaded_file is not None:
    if st.button("🚀 Datei verarbeiten und CSV generieren"):
        if not api_key:
            st.error("❌ Bitte gib zuerst deinen OpenAI API Schlüssel in der Seitenleiste ein.")
        else:
            # Initialize the OpenAI client inside the button click
            try:
                client = OpenAI(api_key=api_key)
                # Simple test call to check authentication (optional but recommended)
                client.models.list()
                st.success("✅ OpenAI API Schlüssel erfolgreich verifiziert.")
            except Exception as e:
                st.error(f"❌ Fehler bei der Initialisierung/Verifizierung des OpenAI Clients: {e}")
                st.stop() # Stop execution if client fails

            start_time = time.time()
            df_output = pd.DataFrame() # Initialize empty DataFrame

            with st.status("Verarbeite Datei...", expanded=True) as status:
                all_phrases = []
                try:
                    # Read content from uploaded file
                    stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
                    all_phrases = stringio.read().splitlines()
                    st.write(f"📄 Erfolgreich {len(all_phrases)} Zeilen aus '{uploaded_file.name}' geladen.")
                except Exception as e:
                    st.error(f"!! Fehler beim Lesen der Datei '{uploaded_file.name}': {e}")
                    status.update(label="Fehler beim Lesen", state="error")
                    st.stop() # Stop if file reading fails

                # 2. Pre-filter Phrases
                st.write(f"🔍 Filtere Phrasen auf {MIN_WORDS}-{MAX_WORDS} Wörter...")
                filtered_phrases, ignored_count = filter_by_word_count(all_phrases, MIN_WORDS, MAX_WORDS)
                st.write(f"   Gefilterte Phrasenanzahl: {len(filtered_phrases)} ({ignored_count} Zeilen ignoriert/nicht passend).")

                if not filtered_phrases:
                    st.warning("Keine Phrasen entsprechen den Wortanzahlkriterien. Verarbeitung gestoppt.")
                    status.update(label="Keine passenden Phrasen", state="error")
                    st.stop()

                # Adjust target phrase count if fewer are available
                current_target_total = TARGET_TOTAL_PHRASES
                if len(filtered_phrases) < current_target_total:
                    st.warning(f"Warnung: Nur {len(filtered_phrases)} passende Phrasen gefunden (< {current_target_total}). Ziel wird auf {len(filtered_phrases)} reduziert.")
                    current_target_total = len(filtered_phrases)

                if current_target_total == 0:
                     st.warning("Zielanzahl der Phrasen ist 0. Verarbeitung gestoppt.")
                     status.update(label="Keine Zielphrasen", state="error")
                     st.stop()

                # 3. Determine Chunking Strategy
                total_chunks = (len(filtered_phrases) + CHUNK_SIZE - 1) // CHUNK_SIZE
                if total_chunks == 0:
                    st.warning("Keine Chunks zum Verarbeiten nach Filterung.")
                    status.update(label="Fehler bei Chunking", state="error")
                    st.stop()

                target_phrases_per_chunk = current_target_total // total_chunks
                remainder = current_target_total % total_chunks
                st.write(f"🧱 Verarbeite in {total_chunks} Chunks (Ziel: max {current_target_total} Phrasen).")

                # 4. Process Chunks to Select High-Quality Phrases
                st.write(f"🤖 Wähle max. {current_target_total} relevante & einzigartige Phrasen via API...")
                all_selected_phrases = []
                api_errors_occurred = False
                for i in range(total_chunks):
                    start_idx = i * CHUNK_SIZE
                    end_idx = min((i + 1) * CHUNK_SIZE, len(filtered_phrases))
                    chunk = filtered_phrases[start_idx:end_idx]
                    chunk_num = i + 1
                    phrases_to_select_this_chunk = target_phrases_per_chunk + (1 if i < remainder else 0)

                    if phrases_to_select_this_chunk == 0: continue

                    st.write(f"   Verarbeite Chunk {chunk_num}/{total_chunks} ({len(chunk)} Phrasen)... Fordere {phrases_to_select_this_chunk} an.")
                    selected_from_chunk, chunk_status_msg = process_chunk(client, chunk, chunk_num, total_chunks, phrases_to_select_this_chunk, existing_selection=all_selected_phrases)

                    if "Fehler" in chunk_status_msg:
                        st.warning(f"   Problem in Chunk {chunk_num}: {chunk_status_msg}")
                        api_errors_occurred = True
                    elif chunk_status_msg: # Display warnings if any
                         st.info(f"   Info Chunk {chunk_num}: {chunk_status_msg}")

                    st.write(f"   -> Erhalten: {len(selected_from_chunk)} Phrasen aus Chunk {chunk_num}.")

                    if len(selected_from_chunk) < phrases_to_select_this_chunk and not api_errors_occurred:
                         st.info(f"   Hinweis: Weniger Phrasen als angefordert ({len(selected_from_chunk)}/{phrases_to_select_this_chunk}) für Chunk {chunk_num} erhalten.")

                    all_selected_phrases.extend(selected_from_chunk)
                    st.write(f"   -> Bisher gesammelt: {len(all_selected_phrases)} Phrasen.")
                    # Optional: time.sleep(1) # Wenn Ratenbegrenzungen auftreten

                # 5. Final Cleanup: Deduplicate and Truncate
                st.write("🧹 Bereinige finale Phrasenauswahl...")
                final_phrases_dedup = list(dict.fromkeys(all_selected_phrases)) # Order-preserving deduplication
                duplicates_removed = len(all_selected_phrases) - len(final_phrases_dedup)
                if duplicates_removed > 0: st.write(f"   {duplicates_removed} exakte Duplikate entfernt.")

                if len(final_phrases_dedup) > current_target_total:
                    st.write(f"   Auswahl ({len(final_phrases_dedup)}) wird auf Ziel {current_target_total} gekürzt.")
                    final_phrases = final_phrases_dedup[:current_target_total]
                else:
                    final_phrases = final_phrases_dedup

                # Final validation for word count
                final_phrases_validated = []
                removed_invalid = 0
                for phrase in final_phrases:
                     word_count = len(phrase.split())
                     if MIN_WORDS <= word_count <= MAX_WORDS:
                         final_phrases_validated.append(phrase)
                     else:
                         st.warning(f"   Warnung (Final Check): Entferne Phrase '{phrase}' ({word_count} Wörter).")
                         removed_invalid += 1
                if removed_invalid > 0:
                    st.write(f"   {removed_invalid} Phrasen im finalen Check wegen Wortanzahl entfernt.")
                final_phrases = final_phrases_validated

                st.write(f"✨ Finale Anzahl Phrasen für Gewichtung/Färbung: {len(final_phrases)}")

                if not final_phrases:
                    st.error("Keine Phrasen nach Auswahlprozess übrig. CSV wird nicht erstellt.")
                    status.update(label="Keine Phrasen ausgewählt", state="error")
                    st.stop()

                # 6. Assign Weights and Colors
                st.write("⚖️ Weise Gewichte via API zu...")
                assigned_weights, weight_status_msg = get_gpt4o_weights(client, final_phrases)
                if "Fehler" in weight_status_msg or "Warnung" in weight_status_msg:
                    st.warning(f"   Problem bei Gewichtung: {weight_status_msg}")
                elif weight_status_msg:
                    st.info(f"   Info Gewichtung: {weight_status_msg}")
                st.write(f"   {len(assigned_weights)} Gewichte zugewiesen.")

                st.write("🎨 Weise Farben zu...")
                assigned_colors = assign_suprima_colors(len(final_phrases))
                st.write(f"   {len(assigned_colors)} Farben zugewiesen.")

                # 7. Create DataFrame
                st.write("📊 Erstelle DataFrame...")
                output_data = {
                    'weight': assigned_weights,
                    'word': final_phrases,
                    'color': assigned_colors,
                    'url': [''] * len(final_phrases) # Add empty URL column
                }
                try:
                    df_output = pd.DataFrame(output_data)
                    # Ensure desired column order
                    df_output = df_output[['weight', 'word', 'color', 'url']]
                    st.write(f"   DataFrame mit {len(df_output)} Zeilen erstellt.")
                except Exception as e:
                     st.error(f"!! Fehler beim Erstellen des DataFrames: {e}")
                     status.update(label="Fehler bei DataFrame", state="error")
                     st.stop()

                end_time = time.time()
                status.update(label=f"Verarbeitung abgeschlossen in {end_time - start_time:.2f}s", state="complete")

            # --- Ergebnisse anzeigen und Download anbieten ---
            st.subheader("Vorschau der generierten Daten")
            st.dataframe(df_output.head())

            st.subheader("Statistiken der finalen Phrasen")
            if not df_output.empty:
                st.write(f"Anzahl finaler Phrasen im CSV: {len(df_output)}")
                word_counts = [len(str(phrase).split()) for phrase in df_output['word']]
                stats_data = []
                for i in range(MIN_WORDS, MAX_WORDS + 1):
                    count = word_counts.count(i)
                    percentage = (count / len(df_output)) * 100 if len(df_output) > 0 else 0
                    stats_data.append({
                        "Wortanzahl": f"{i} Wort{'e' if i > 1 else ''}",
                        "Anzahl Phrasen": count,
                        "Prozent": f"{percentage:.1f}%"
                    })
                st.table(pd.DataFrame(stats_data))
            else:
                st.info("Keine finalen Phrasen zum Anzeigen von Statistiken vorhanden.")


            # --- Download Button ---
            st.subheader("Download")
            try:
                # Konvertiere DataFrame zu CSV-String im Speicher
                csv_data = df_output.to_csv(index=False, encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC).encode('utf-8')

                output_filename = f"wordcloud_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"

                st.download_button(
                    label="💾 CSV-Datei herunterladen",
                    data=csv_data,
                    file_name=output_filename,
                    mime='text/csv',
                 )
                st.success(f"Datei '{output_filename}' zum Download bereit!")
            except Exception as e:
                st.error(f"Fehler beim Vorbereiten der CSV für den Download: {e}")

# Zeige Anleitung, wenn keine Datei hochgeladen wurde
if uploaded_file is None:
    st.info("Bitte lade eine .txt-Datei hoch, um zu beginnen.")