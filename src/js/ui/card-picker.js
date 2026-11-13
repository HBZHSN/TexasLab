import { CARDS_BY_CODE, DECK } from "../core/cards.js";
import { RANK_TEXT, STREET_BOARD_COUNTS, SUITS } from "../core/constants.js";
import { byId, selectAll } from "./dom.js";

export class CardPicker {
  constructor(state) {
    this.state = state;
  }

  enabledSlots() {
    return [
      "h0",
      "h1",
      ...Array.from(
        { length: STREET_BOARD_COUNTS[this.state.street] },
        (_, index) => `b${index}`,
      ),
    ];
  }

  knownCards() {
    const hero = [this.state.slots.h0, this.state.slots.h1].filter(
      (code) => code !== null,
    );
    const board = Array.from(
      { length: STREET_BOARD_COUNTS[this.state.street] },
      (_, index) => this.state.slots[`b${index}`],
    ).filter((code) => code !== null);
    return { hero, board, all: [...hero, ...board] };
  }

  trimBoard() {
    for (let index = STREET_BOARD_COUNTS[this.state.street]; index < 5; index += 1) {
      this.state.slots[`b${index}`] = null;
    }
  }

  clear() {
    Object.keys(this.state.slots).forEach((slot) => {
      this.state.slots[slot] = null;
    });
    this.state.activeSlot = "h0";
    this.render();
  }

  setCards(hero, board = []) {
    Object.keys(this.state.slots).forEach((slot) => {
      this.state.slots[slot] = null;
    });
    hero.forEach((code, index) => {
      this.state.slots[`h${index}`] = code;
    });
    board.forEach((code, index) => {
      this.state.slots[`b${index}`] = code;
    });
    this.state.activeSlot = this.nextSlot("h1");
    this.render();
  }

  nextSlot(currentSlot) {
    const enabled = this.enabledSlots();
    const currentIndex = enabled.indexOf(currentSlot);
    for (let offset = 1; offset <= enabled.length; offset += 1) {
      const slot = enabled[(currentIndex + offset) % enabled.length];
      if (this.state.slots[slot] === null) return slot;
    }
    return enabled[(currentIndex + 1) % enabled.length];
  }

  assign(code) {
    Object.keys(this.state.slots).forEach((slot) => {
      if (this.state.slots[slot] === code) this.state.slots[slot] = null;
    });
    this.state.slots[this.state.activeSlot] = code;
    this.state.activeSlot = this.nextSlot(this.state.activeSlot);
    this.render();
  }

  render() {
    this.renderSlots();
    this.renderDeck();
  }

  renderDeck() {
    const used = new Set(
      Object.values(this.state.slots).filter((code) => code !== null),
    );
    byId("deck").innerHTML = DECK.map(
      (card) =>
        `<button class="deck-card ${card.red ? "red" : ""}" data-card="${card.code}" ${used.has(card.code) ? "disabled" : ""}>${card.text}</button>`,
    ).join("");
    selectAll(".deck-card").forEach((button) => {
      button.onclick = () => this.assign(Number(button.dataset.card));
    });
  }

  renderSlots() {
    const enabled = new Set(this.enabledSlots());
    selectAll(".card-slot").forEach((element) => {
      const slot = element.dataset.slot;
      const code = this.state.slots[slot];
      const card = CARDS_BY_CODE.get(code);
      const isEnabled = enabled.has(slot);
      element.className = [
        "card-slot",
        this.state.activeSlot === slot && isEnabled ? "active" : "",
        card && isEnabled ? "filled" : "",
        card?.red && isEnabled ? "red" : "",
        isEnabled ? "" : "hidden",
      ]
        .filter(Boolean)
        .join(" ");

      if (card && isEnabled) {
        element.innerHTML = `<span class="rank">${RANK_TEXT[card.rank]}</span><span class="suit">${SUITS[card.suit].symbol}</span><button class="remove" aria-label="移除 ${card.text}">×</button>`;
        element.querySelector(".remove").onclick = (event) => {
          event.stopPropagation();
          this.state.slots[slot] = null;
          this.state.activeSlot = slot;
          this.render();
        };
      } else {
        element.innerHTML = '<span class="slot-placeholder">+</span>';
      }
      if (isEnabled) {
        element.onclick = () => {
          this.state.activeSlot = slot;
          this.renderSlots();
        };
      }
    });
  }
}
