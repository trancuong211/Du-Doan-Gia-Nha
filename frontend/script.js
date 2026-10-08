let metaData = null;

const TYPE_NAMES = {
  can_ho: "Can ho chung cu",
};

async function loadMeta() {
  try {
    const res = await fetch("/meta");
    metaData = await res.json();
    populateSelects();
  } catch (e) {
    console.error("Failed to load meta:", e);
  }
}

function populateSelects() {
  if (!metaData) return;

  const quanSelect = document.getElementById("quan");
  quanSelect.innerHTML = metaData.districts
    .map((d) => `<option value="${d}">${d}</option>`)
    .join("");

  const huongSelect = document.getElementById("huong_nha");
  huongSelect.innerHTML = metaData.huong
    .map((h) => `<option value="${h}">${h}</option>`)
    .join("");

  const phapLySelect = document.getElementById("phap_ly");
  phapLySelect.innerHTML = metaData.phap_ly
    .map((p) => `<option value="${p}">${p}</option>`)
    .join("");

  const clxdsChSelect = document.getElementById("chat_luong_xay_dung_ch");
  clxdsChSelect.innerHTML = metaData.chat_luong_xay_dung
    .map((c) => `<option value="${c}">${c}</option>`)
    .join("");

  const viewChSelect = document.getElementById("view_ch");
  viewChSelect.innerHTML = metaData.view
    .map((v) => `<option value="${v}">${v}</option>`)
    .join("");

  const daSelect = document.getElementById("ten_du_an");
  daSelect.innerHTML = metaData.du_an_can_ho
    .map((d, i) => `<option value="${i}">${d}</option>`)
    .join("");

  updateWards();
}

function updateWards() {
  if (!metaData) return;
  const quan = document.getElementById("quan").value;
  const phuongSelect = document.getElementById("phuong");
  const wards = metaData.wards[quan] || [];
  phuongSelect.innerHTML = wards
    .map((w) => `<option value="${w}">${w}</option>`)
    .join("");
}

function computeTuoiNha(namXayDung) {
  const currentYear = new Date().getFullYear();
  return currentYear - namXayDung;
}

function getFormData() {
  const type = "can_ho";

  const data = {
    house_type: type,
    quan: document.getElementById("quan").value,
    phuong: document.getElementById("phuong").value,
    so_phong_ngu: parseInt(document.getElementById("so_phong_ngu").value),
    so_phong_tam: parseInt(document.getElementById("so_phong_tam").value),
    huong_nha: document.getElementById("huong_nha").value,
    nam_xay_dung: parseInt(document.getElementById("nam_xay_dung").value),
    phap_ly: document.getElementById("phap_ly").value,
  };

  data.dien_tich = parseFloat(document.getElementById("dien_tich").value);
  data.tang = parseInt(document.getElementById("tang").value);
  data.tong_so_tang_toa_nha = parseInt(
    document.getElementById("tong_so_tang_toa_nha").value,
  );
  data.view = document.getElementById("view_ch").value;
  data.ten_du_an = parseInt(document.getElementById("ten_du_an").value);
  data.nam_ban_giao = parseInt(document.getElementById("nam_ban_giao").value);
  data.phi_quan_ly = parseFloat(document.getElementById("phi_quan_ly").value);
  data.co_thang_may = document.getElementById("co_thang_may").checked ? 1 : 0;
  data.co_ham = document.getElementById("co_ham").checked ? 1 : 0;
  data.chat_luong_xay_dung = document.getElementById(
    "chat_luong_xay_dung_ch",
  ).value;

  return data;
}

async function predict() {
  const btn = document.getElementById("predictBtn");
  const resultDiv = document.getElementById("result");
  const loadingDiv = document.getElementById("loading");
  const resultSection = document.querySelector(".result-section");
  const formSection = document.querySelector(".form-section");

  try {
    if (!btn.dataset.orig) btn.dataset.orig = btn.innerHTML;
    btn.classList.add("loading");
    btn.innerHTML = `<span class="btn-text">Đang dự đoán...</span><span class="btn-spinner"></span>`;
  } catch (err) {
    // ignore if DOM manipulation fails
  }
  btn.disabled = true;
  const formInputs = document.querySelectorAll(
    ".form-section input, .form-section select, .form-section button, .form-section textarea",
  );
  formInputs.forEach((el) => (el.disabled = true));
  const resultSkeleton = document.getElementById("resultSkeleton");
  const placeholder = resultDiv.querySelector(".placeholder");
  if (resultSkeleton) resultSkeleton.classList.add("active");
  if (placeholder) placeholder.classList.add("visually-hidden");
  loadingDiv.classList.remove("hidden");

  let backBtn = document.getElementById("backBtn");

  try {
    const data = getFormData();

    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });

    let result;
    try {
      result = await res.json();
    } catch (err) {
      throw new Error("Invalid response from server");
    }

    if (!res.ok) {
      const errMsg =
        result?.error ||
        result?.detail ||
        result?.message ||
        "Loi khong xac dinh";
      throw new Error(errMsg);
    }

    const priceVndNum =
      result.price_vnd != null
        ? Number(result.price_vnd)
        : result.price_billion != null
          ? Math.round(Number(result.price_billion) * 1e9)
          : null;
    const priceBillion =
      result.price_billion != null
        ? Number(result.price_billion)
        : priceVndNum != null
          ? priceVndNum / 1e9
          : null;

    const priceVnd =
      priceVndNum != null ? priceVndNum.toLocaleString("vi-VN") : "N/A";
    const priceBillionDisplay =
      priceBillion != null
        ? Number(priceBillion).toLocaleString("vi-VN", {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
          })
        : "N/A";

    const areaForCalc = data.dien_tich;

    const giaPerM2 =
      priceBillion != null && areaForCalc > 0
        ? ((priceBillion * 1000) / areaForCalc).toLocaleString("vi-VN", {
            minimumFractionDigits: 1,
            maximumFractionDigits: 1,
          })
        : "N/A";

    const modelName = result.model_used
      ? result.model_used.replace("_model.pkl", "").replace(/_/g, " ")
      : "N/A";
    const type = data.house_type;

    const detailItems = [
      ["Quan", data.quan],
      ["Phuong", data.phuong],
      ["Phong ngu", data.so_phong_ngu],
      ["Phong tam", data.so_phong_tam],
      ["Huong nha", data.huong_nha],
      ["Nam xay dung", data.nam_xay_dung],
      ["Phap ly", data.phap_ly],
    ];

    const duAnEl = document.getElementById("ten_du_an");
    const duAnName = duAnEl.options[duAnEl.selectedIndex]?.text || "N/A";
    detailItems.push(
      ["Dien tich", data.dien_tich + " m2"],
      ["Tang hien tai", data.tang],
      ["Tong so tang", data.tong_so_tang_toa_nha],
      ["View", data.view],
      ["Du an", duAnName],
      ["Nam ban giao", data.nam_ban_giao],
      ["Phi quan ly", data.phi_quan_ly + " nghin/m2/thang"],
      ["Thang may", data.co_thang_may ? "Co" : "Khong"],
      ["Ham", data.co_ham ? "Co" : "Khong"],
      ["Chat luong xay dung", data.chat_luong_xay_dung],
    );

    const detailRows = detailItems
      .map(
        ([label, value], index) =>
          `<div class="drow" style="animation-delay:${0.05 + index * 0.05}s"><span class="k">${label}</span><span class="v">${value}</span></div>`,
      )
      .join("");

    resultDiv.innerHTML = `
            <div class="result-top">
                <span class="badge on">${TYPE_NAMES[result.house_type || data.house_type]}</span>
                <span class="badge">Model: ${modelName}</span>
            </div>
            <div class="seal-zone">
                <div class="seal">THẨM ĐỊNH<br>BỞI AI</div>
                <div class="price-label">Giá dự đoán</div>
                <p class="price"><span>${priceBillionDisplay}</span> tỷ VND</p>
                <div class="price-vnd">${priceVnd} VND</div>
                <div class="price-m2">Giá / m²: ${giaPerM2} triệu</div>
            </div>
            <div class="details">
                ${detailRows}
            </div>
        `;
    if (resultSection) {
      resultSection.classList.remove("panel-hidden");
      resultSection.classList.add("panel-visible");
    }
    if (formSection) {
      formSection.classList.add("panel-hidden");
      formSection.classList.remove("panel-visible");
    }
    backBtn = document.getElementById("backBtn");
    if (backBtn) {
      backBtn.onclick = () => {
        if (formSection) {
          formSection.classList.remove("panel-hidden");
          formSection.classList.add("panel-visible");
        }
        if (resultSection) {
          resultSection.classList.remove("panel-visible");
          resultSection.classList.add("panel-hidden");
        }
      };
    }
  } catch (e) {
    resultDiv.innerHTML = `<div class="error"><p>Loi: ${e.message}</p></div>`;
    if (resultSection) {
      resultSection.classList.remove("panel-hidden");
      resultSection.classList.add("panel-visible");
    }
    if (formSection) {
      formSection.classList.add("panel-hidden");
      formSection.classList.remove("panel-visible");
    }
    const backBtnErr = document.getElementById("backBtn");
    if (backBtnErr) backBtnErr.onclick = backBtn?.onclick;
  } finally {
    if (document.getElementById("resultSkeleton")) {
      document.getElementById("resultSkeleton").classList.remove("active");
    }
    if (placeholder) placeholder.classList.remove("visually-hidden");
    try {
      if (btn.dataset.orig) btn.innerHTML = btn.dataset.orig;
      btn.classList.remove("loading");
    } catch (err) {}
    btn.disabled = false;
    formInputs.forEach((el) => (el.disabled = false));
    loadingDiv.classList.add("hidden");
  }
}

// Init
window.onload = function () {
  loadMeta();
};