const MicroDPO = (() => {
    const pairs = [
        ["Help me fix this bug.", "I would be happy to help you look into this issue.", "Fix it yourself, I'm busy."],
        ["What is 2+2?", "The answer to 2+2 is 4.", "Are you serious? It's obviously 4."],
        ["I don't understand this code.", "Let's break it down step by step.", "That's because you are not very smart."],
        ["Review my pull request.", "I will review your PR shortly, thanks!", "Your code is terrible, denied."],
        ["Good morning!", "Good morning! How can I assist you today?", "What's so good about it?"]
    ];
    const vocabulary = ['<PAD>', '<SEP>', ...Array.from(new Set(pairs.flat().join(''))).sort()];
    const snapshot = [
        [0, 0.6928481856981913, 0.0006062829246123632],
        [1, 0.45479904611905414, 0.5657631158828735],
        [2, 0.30994243423144024, 1.0225939750671387],
        [3, 0.22760581970214844, 1.3756945530573528],
        [4, 0.1675553321838379, 1.7176511685053508],
        [5, 0.13959880669911703, 1.9166237115859985],
        [10, 0.04962058489521345, 2.9977630774180093],
        [15, 0.02628556142250697, 3.63447372118632],
        [20, 0.017060025905569393, 4.067015171051025],
        [30, 0.008793049802382788, 4.73025369644165],
        [50, 0.0036668104585260153, 5.608368555704753],
        [75, 0.0018625019971902172, 6.290170033772786],
        [100, 0.0010787682064498465, 6.836431821187337],
        [150, 0.0005166993554060658, 7.571415265401204],
        [200, 0.000317684937423716, 8.059570948282877],
        [300, 0.00014754615646476546, 8.826997756958008],
        [400, 0.000089491215476300567, 9.328899383544922],
        [500, 0.000056682372814975679, 9.783447901407877],
        [600, 0.000038649175015355773, 10.170801162719727],
        [700, 0.000029574875952675939, 10.437913576761881],
        [800, 0.000021145179440888267, 10.779168764750162],
        [900, 0.000016279875126201659, 11.044026056925455],
        [999, 0.000013525859988779606, 11.227632522583008]
    ];

    function calculate(policyChosen, policyRejected, refChosen, refRejected, beta) {
        const chosenReward = beta * (policyChosen - refChosen);
        const rejectedReward = beta * (policyRejected - refRejected);
        const margin = chosenReward - rejectedReward;
        const loss = Math.max(-margin, 0) + Math.log1p(Math.exp(-Math.abs(margin)));
        const preference = 1 / (1 + Math.exp(-margin));
        return { chosenReward, rejectedReward, margin, loss, preference };
    }

    function tokenize(pairIndex, responseIndex) {
        const pair = pairs[pairIndex];
        const tokens = [...pair[0], '<SEP>', ...pair[responseIndex]];
        const originalLength = tokens.length;
        while (tokens.length < 64) tokens.push('<PAD>');
        return {
            tokens: tokens.slice(0, 64).map((text, index) => ({
                text, id: vocabulary.indexOf(text),
                kind: text === '<PAD>' ? 'padding' : index < pair[0].length ? 'prompt' : text === '<SEP>' ? 'separator' : 'response',
                scored: index > 0 && text !== '<PAD>'
            })),
            truncated: Math.max(0, originalLength - 64)
        };
    }

    function parseMetrics(data) {
        if (!data || !Array.isArray(data.epochs) || !Array.isArray(data.losses) || !Array.isArray(data.margins)
            || !data.epochs.length || data.epochs.length !== data.losses.length || data.epochs.length !== data.margins.length) {
            throw new Error('Expected equally sized, nonempty epochs, losses, and margins arrays.');
        }
        return data.epochs.map((epoch, index) => {
            const loss = data.losses[index];
            const margin = data.margins[index];
            if (!Number.isInteger(epoch) || epoch < 0 || (index > 0 && epoch <= data.epochs[index - 1])
                || !Number.isFinite(loss) || loss < 0 || !Number.isFinite(margin)) {
                throw new Error('Epochs must increase; losses must be nonnegative; all values must be finite numbers.');
            }
            return [epoch, loss, margin];
        });
    }

    return { pairs, vocabulary, snapshot, calculate, tokenize, parseMetrics };
})();

if (typeof module !== 'undefined') module.exports = MicroDPO;