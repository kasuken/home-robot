"""Games to play together (the owner's ask): "Laika, giochiamo?".

`GAMES` is the catalogue. Most games are played in words and the model runs them from the rules that
`start_game` hands it (riddles, a quiz, guess what I'm thinking, a word chain, a story together, "Laika
dice", the colour hunt with the camera, animal sounds). Three use the body, and a `play_round` call runs
one round of them in code, deterministically, so the model never has to time anything:

* "Un, due, tre... stella!": she turns her head away and counts, the child creeps closer; at "stella!" she
  looks back and reads the ultrasonic sensor for `FREEZE_S`. A reading that changes by more than `MOVE_CM`
  means the child moved ("ti ho visto!"); closer than `WIN_CM` and the child has won. The ultrasonic sensor
  is on the chassis, so it measures the child whichever way the head points.
* Statues ("statue"): she dances (head and steering only, the wheels never turn), then says "statua!" and
  checks the same way that nobody moves.
* Hide and seek ("nascondino"): she looks down and counts to ten aloud, then looks for a person with the
  searcher (driving only where autonomous driving is allowed, else a head scan) and says what she found.

The distance comes from the safety monitor's latest reading (it reads the sensors 20 times a second), so a
game never touches the ultrasonic sensor itself. While a game is on, the conversation waits longer for an
answer (`FOLLOW_UP_S`): children think. The game ends with `stop_game` or when the conversation ends.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from laika.limits import PAN_MAX, TILT_MIN

if TYPE_CHECKING:
    from laika.actions import ActionPlayer
    from laika.robot import Robot
    from laika.safety import SafetyController
    from laika.search import Searcher

# The conversation waits this long for an answer while a game is on.
FOLLOW_UP_S = 30.0
FREEZE_S = 2.5  # how long she watches after "stella!" / "statua!"
SAMPLE_S = 0.1
MOVE_CM = 12.0  # change in distance that counts as moving (the sensor jitters a few cm)
WIN_CM = 25.0  # closer than this at "stella!" and the child has reached her
MIN_SAMPLES = 5
DANCE_S = 5.0
SENSOR_MAX_AGE_S = 0.5
SEEK_MAX_S = 120.0
COUNT_TO = 10


@dataclass(frozen=True)
class Game:
    id: str
    name: str  # Italian, as she says it
    kind: str  # "voice" (the model runs it) or "body" (play_round runs each round)
    rules: str  # Italian, for the model
    needs: str = ""  # "camera", "search"


GAMES: tuple[Game, ...] = (
    Game(
        "indovinelli", "Indovinelli", "voice",
        "Fai un indovinello adatto a un bambino, uno alla volta. Aspetta la risposta; se sbaglia dai un "
        "indizio (al massimo due), poi dici la soluzione. Festeggia (gesture celebrate) quando indovina. "
        "Ogni tanto chiedi se vuole proporne uno lui a te.",
    ),
    Game(
        "quiz", "Quiz", "voice",
        "Un quiz di cinque domande facili su animali, spazio, natura o corpo umano (chiedi prima il tema). "
        "Una domanda alla volta, con tre risposte possibili. Tieni il punteggio e dillo alla fine. "
        "Sii incoraggiante quando sbaglia.",
    ),
    Game(
        "indovina_cosa_penso", "Indovina cosa penso", "voice",
        "Pensa a un animale o a un oggetto di casa e non dirlo. Il bambino fa domande a cui rispondi solo "
        "sì o no; dopo dieci domande dai un indizio. Poi a ruoli invertiti: pensa lui, e tu fai domande "
        "sì/no per indovinare.",
    ),
    Game(
        "catena_di_parole", "Catena di parole", "voice",
        "A turno dite una parola che comincia con l'ultima sillaba della parola di prima (gatto, tobia, "
        "biagio...). Niente parole già dette. Se lui si blocca, aiutalo con un suggerimento.",
    ),
    Game(
        "storia_insieme", "Inventiamo una storia", "voice",
        "Inventate una storia insieme, una o due frasi a testa. Comincia tu con un inizio buffo (magari "
        "con una cagnolina nello spazio). Riprendi quello che dice lui e aggiungi una sorpresa. Dopo una "
        "decina di turni trova un bel finale.",
    ),
    Game(
        "laika_dice", "Laika dice", "voice",
        "Come \"Simone dice\": dai ordini semplici (tocca il naso, salta, gira su te stesso). Va fatto "
        "solo se l'ordine comincia con \"Laika dice\"; ogni tanto prova a ingannarlo senza. Poi a ruoli "
        "invertiti: lui ti dà ordini e tu li esegui con i gesti e gli sguardi (gesture, look_*) solo se "
        "cominciano con \"Laika dice\", altrimenti dici \"ah, non ci casco!\". Mai le ruote per questo gioco.",
    ),
    Game(
        "caccia_al_colore", "Caccia al colore", "voice",
        "Chiedi di portarti qualcosa di un colore (o una forma: rotondo, morbido...). Quando dice che è "
        "pronto, guarda con capture_image e controlla nella descrizione se c'è davvero. Un punto per ogni "
        "oggetto giusto, cambia colore ogni volta.",
        needs="camera",
    ),
    Game(
        "versi_degli_animali", "Indovina il verso", "voice",
        "Fai il verso di un animale a parole (\"muuu\", \"bau bau\", \"cra cra\") e lui indovina quale. Ogni "
        "tanto descrivi l'animale invece del verso. Poi tocca a lui fare il verso.",
    ),
    Game(
        "un_due_tre_stella", "Un, due, tre... stella!", "body",
        "Spiega: lui si mette lontano, davanti a te; mentre conti si avvicina, a \"stella\" deve stare "
        "fermo. Ogni round è una chiamata a play_round: ti giri, conti, ti volti e controlli col sensore. "
        "Dopo il risultato commenta (\"ti ho visto!\", \"bravo, immobile!\") e chiedi se è pronto per il "
        "prossimo. Vince quando arriva vicinissimo a te.",
    ),
    Game(
        "statue", "Statue", "body",
        "Spiega: mentre balli, lui balla; quando dici \"statua!\" deve restare immobile davanti a te. Ogni "
        "round è una chiamata a play_round. Dopo il risultato commenta e chiedi se vuole un altro round.",
    ),
    Game(
        "nascondino", "Nascondino", "body",
        "Spiega: tu conti fino a dieci guardando in basso, lui si nasconde in questa stanza, poi lo cerchi. "
        "Quando è pronto chiama play_round. Se non lo trovi, chiedi di farsi vedere e fagli i complimenti.",
        needs="search",
    ),
)
BY_ID = {game.id: game for game in GAMES}


class GameSession:
    """The game being played, if any. Safe from any thread."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._game: Game | None = None
        self.rounds = 0
        self.score = {"child": 0, "laika": 0}

    @property
    def game(self) -> Game | None:
        with self._lock:
            return self._game

    @property
    def active(self) -> bool:
        return self.game is not None

    def start(self, game_id: str) -> Game:
        game = BY_ID[game_id]
        with self._lock:
            self._game = game
            self.rounds = 0
            self.score = {"child": 0, "laika": 0}
        print(f"[laika] game: {game.name}")
        return game

    def end(self) -> Game | None:
        with self._lock:
            game, self._game = self._game, None
        if game is not None:
            print(f"[laika] game over: {game.name} ({self.rounds} rounds)")
        return game

    def follow_up_s(self, default: float) -> float:
        return max(default, FOLLOW_UP_S) if self.active else default


class GameRounds:
    """Runs one round of a body game. Blocks the caller (the agent's turn) while it runs."""

    def __init__(
        self,
        robot: Robot,
        safety: SafetyController,
        speak: Callable[[str], None],
        *,
        actions: ActionPlayer | None = None,
        searcher: Searcher | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
    ) -> None:
        self.robot = robot
        self.safety = safety
        self.speak = speak
        self.actions = actions
        self.searcher = searcher
        self._sleep = sleep
        self._clock = clock
        self._rng = rng or random.Random()

    def play(self, game: Game, *, lease: Any = None, should_stop: Callable[[], bool] | None = None) -> dict[str, Any]:
        if game.id == "un_due_tre_stella":
            return self.red_light(should_stop)
        if game.id == "statue":
            return self.statues(should_stop)
        if game.id == "nascondino":
            return self.hide_and_seek(lease, should_stop)
        return {"ok": False, "error": f"{game.name} si gioca a parole: niente play_round"}

    # --- the rounds ---------------------------------------------------------------

    def red_light(self, should_stop: Callable[[], bool] | None = None) -> dict[str, Any]:
        self.robot.look(PAN_MAX, 0)  # her back to the child
        # A different rhythm each round, so he cannot time it.
        self.speak(self._rng.choice(("Un, due, tre...", "Uuun... due... tre...", "Un-due-tre...")))
        self._sleep(self._rng.uniform(0.0, 1.5))
        self.robot.look(0, 0)
        self.speak("Stella!")
        return self._watch("stella", should_stop, win=True)

    def statues(self, should_stop: Callable[[], bool] | None = None) -> dict[str, Any]:
        if self.actions is not None:
            self.actions.play("dance")
        self._sleep(self._rng.uniform(DANCE_S - 2, DANCE_S + 2))
        if self.actions is not None:
            self.actions.cancel_pending()
        self.robot.look(0, 0)
        self.speak("Statua!")
        return self._watch("statue", should_stop, win=False)

    def hide_and_seek(self, lease: Any, should_stop: Callable[[], bool] | None = None) -> dict[str, Any]:
        if self.searcher is None:
            return {"ok": False, "error": "non posso cercare persone qui: gioca a un altro gioco"}
        self.robot.look(0, TILT_MIN)  # eyes covered
        self.speak(", ".join(str(n) for n in range(1, COUNT_TO + 1)) + "! Pronto o no, arrivo!")
        self.robot.look(0, 0)
        if should_stop is not None and should_stop():
            return {"ok": False, "error": "interrupted"}
        found = self.searcher.find_person(lease=lease, max_s=SEEK_MAX_S, should_stop=should_stop)
        result = {"ok": True, "game": "nascondino", "found": bool(found.get("found"))}
        for key in ("where", "how", "error"):
            if key in found:
                result[key] = found[key]
        result["say"] = "trovato! Festeggia" if result["found"] else "non l'hai trovato: chiedigli di farsi vedere"
        return result

    # --- watching with the ultrasonic sensor ---------------------------------------------

    def _watch(self, game: str, should_stop: Callable[[], bool] | None, *, win: bool) -> dict[str, Any]:
        readings: list[float] = []
        end = self._clock() + FREEZE_S
        while self._clock() < end:
            if should_stop is not None and should_stop():
                return {"ok": False, "error": "interrupted"}
            distance = self._distance()
            if distance is not None:
                readings.append(distance)
            self._sleep(SAMPLE_S)
        result: dict[str, Any] = {"ok": True, "game": game}
        if len(readings) < MIN_SAMPLES:
            result["seen"] = False
            result["say"] = "non ti vedo col sensore: chiedigli di mettersi davanti a te, a meno di due metri"
            return result
        spread = max(readings) - min(readings)
        closest = min(readings)
        result.update(seen=True, distance_cm=round(closest), moved=spread > MOVE_CM, movement_cm=round(spread))
        if win and closest <= WIN_CM:
            result["won"] = True
            result["say"] = "è arrivato fino a te: ha vinto! Festeggia"
        elif result["moved"]:
            result["say"] = "si è mosso: l'hai visto! Deve tornare indietro" if win else "si è mosso: eliminato, ma ridendo"
        else:
            result["say"] = "era immobile: bravissimo"
        return result

    def _distance(self) -> float | None:
        reading = self.safety.latest_sensors
        if reading is None or time.time() - reading.timestamp > SENSOR_MAX_AGE_S:
            return None
        return reading.distance_cm


def catalogue(*, camera: bool, search: bool) -> list[Game]:
    """The games this robot can play (the colour hunt needs the camera, hide and seek the searcher)."""
    out = []
    for game in GAMES:
        if game.needs == "camera" and not camera:
            continue
        if game.needs == "search" and not search:
            continue
        out.append(game)
    return out
